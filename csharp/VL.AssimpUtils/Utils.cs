using System;
using System.Collections.Generic;
using System.Linq;
using Assimp;
using Quaternion = Stride.Core.Mathematics.Quaternion;
#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

namespace VL.AssimpGpu
{
    // -------------------- GPU structs --------------------

    /// Vertex layout for GPU (StructuredBuffer); tightly packed / sequential.
    public struct GpuVertex
    {
        public Vector3 Position;     // 12
        public Vector3 Normal;       // 24
        public Vector2 Tex;          // 32
        public Int4 Bone; // 48
        public Vector4 Weights;      // 64 bytes total
    }

    public struct SkeletonGpu
    {
        public int[] Parent;        // bone -> parent (-1 root)
        public Matrix[] InverseBind;   // bind^-1 (Assimp Bone.OffsetMatrixrix)
        public Matrix[] BindLocal;     // optional (node.Transform), useful if you build bindLocal on GPU
    }

    public struct DualQuat { public Vector4 Qr, Qd; } // (x,y,z,w), (x,y,z,w)

    public struct ClipInfoRaster
    {
        public int BoneCount;
        public int FrameCount;

        // index into LocalDeltaDQ (DualQuat-units)
        public int StartIndex;

        public float TicksPerSecond;
        public float StartTick;
        public float StepTick;
        public float DurationSeconds;
    }

    public sealed class AnimBankGpu
    {
        // delta-local DQs: animLocal = restInvLocal * currentLocal
        public DualQuat[] LocalDeltaDq = [];

        // optional convenience if you ever want ready-to-skin on GPU
        public DualQuat[] SkinDq = [];

        public ClipInfoRaster[] Clips = [];
    }

    public sealed class MeshGpu
    {
        public GpuVertex[] Vertices = [];
        public int[] Indices = []; // 32-bit for simplicity
        public int BoneCount;
    }

    public sealed class ModelGpu
    {
        public SkeletonGpu Skeleton = new();
        public MeshGpu Mesh = new();
        public AnimBankGpu Anim = new();
        // mapping, only for debugging or tools
        public string[] BoneNames = [];
    }

    // -------------------- Loader --------------------

    public static class MixamoGpuLoader
    {
        /// <summary>
        /// Load a single-character Mixamo FBX to GPU-friendly buffers.
        /// - max 4 weights per vertex (normalized)
        /// - per-clip delta-local DQs (for bindLocal*animLocal on GPU)
        /// - optional SkinDQ bank
        /// </summary>
        public static ModelGpu LoadFbx(string path,
                                       float sceneScale = 0.01f,     // Mixamo (cm) -> meters
                                       float fpsIfNeeded = 30f,
                                       float tpsFallback = 25f,
                                       float uniformTolerance = 1e-4f)
        {
            var ctx = new AssimpContext();
            var pps = PostProcessSteps.Triangulate
                    | PostProcessSteps.JoinIdenticalVertices
                    | PostProcessSteps.LimitBoneWeights
                    | PostProcessSteps.ImproveCacheLocality
                   // | PostProcessSteps.FlipWindingOrder // Mixamo often CCW; adjust if needed
                    | PostProcessSteps.OptimizeMeshes
                    | PostProcessSteps.OptimizeGraph;

            var scene = ctx.ImportFile(path, pps);
            if (scene == null || scene.RootNode == null)
                throw new InvalidOperationException("Failed to load scene.");

            // 1) Skeleton (names -> indices, parent, bind, rest)
            // Note: sceneScale is applied explicitly to all translation data below
            var boneNames = GatherBoneNames(scene);
            var nodeByName = new Dictionary<string, Node>(StringComparer.Ordinal);
            Walk(scene.RootNode, n => nodeByName[n.Name] = n);

            var orderedBones = new List<string>();
            Walk(scene.RootNode, n => { if (boneNames.Contains(n.Name)) orderedBones.Add(n.Name); });
            foreach (var bn in boneNames) if (!orderedBones.Contains(bn)) orderedBones.Add(bn); // stragglers

            int nb = orderedBones.Count;
            var indexOf = orderedBones.Select((n, i) => (n, i))
                                      .ToDictionary(t => t.n, t => t.i, StringComparer.Ordinal);

            var parentName = new Dictionary<string, string>(StringComparer.Ordinal);
            Walk(scene.RootNode, n => { foreach (var c in n.Children) parentName[c.Name] = n.Name; });

            var parent = Enumerable.Repeat(-1, nb).ToArray();
            var bindLocal = new Matrix[nb];
            var bindGlobal = new Matrix[nb];
            var invBind = new Matrix[nb];

            // map boneName -> OffsetMatrixrix (bind^-1)
            var invBindByName = new Dictionary<string, Matrix>(StringComparer.Ordinal);
            foreach (var mesh0 in scene.Meshes)
                foreach (var b in mesh0.Bones)
                    invBindByName[b.Name] = ToM(b.OffsetMatrix);

            for (int i = 0; i < nb; i++)
            {
                var name = orderedBones[i];
                var localMatrix = nodeByName.TryGetValue(name, out var nn) ? ToM(nn.Transform) : Matrix.Identity;
                
                // Apply scene scale to translation component
                localMatrix.M41 *= sceneScale;
                localMatrix.M42 *= sceneScale;
                localMatrix.M43 *= sceneScale;
                bindLocal[i] = localMatrix;
                
                // Find nearest ancestor bone by walking up the node hierarchy
                if (nodeByName.TryGetValue(name, out var node) && node.Parent != null)
                {
                    var parentNode = node.Parent;
                    while (parentNode != null)
                    {
                        if (indexOf.TryGetValue(parentNode.Name, out var pi))
                        {
                            parent[i] = pi;
                            break;
                        }
                        parentNode = parentNode.Parent;
                    }
                }
            }

            // Build invBind array 
            for (int i = 0; i < nb; i++)
            {
                var name = orderedBones[i];
                var invBindMatrix = invBindByName.TryGetValue(name, out var m) ? m : Matrix.Identity;
                
                // Apply scene scale to translation component of inverse bind
                invBindMatrix.M41 *= sceneScale;
                invBindMatrix.M42 *= sceneScale;
                invBindMatrix.M43 *= sceneScale;
                invBind[i] = invBindMatrix;
            }
            
            // accumulate global bind (root-first order due to traversal choice)
            for (var i = 0; i < nb; i++)
            {
                var p = parent[i];
                bindGlobal[i] = p < 0 ? bindLocal[i] : Matrix.Multiply(bindGlobal[p], bindLocal[i]);
            }

            var skeleton = new SkeletonGpu
            {
                Parent = parent, 
                InverseBind = invBind, 
                BindLocal = bindLocal
            };

            // 2) Mesh (pick first skinned mesh; Mixamo uses one)
            var (mesh, meshToBone) = BuildMesh(scene, indexOf, sceneScale);

            // 3) AniMatrixions → delta-local DQs per frame
            var anim = BuildAniMatrixions(scene, orderedBones, parent, bindLocal, 
                                       fpsIfNeeded, tpsFallback, uniformTolerance, sceneScale);

            return new ModelGpu
            {
                Skeleton = skeleton,
                Mesh = mesh,
                Anim = anim,
                BoneNames = orderedBones.ToArray()
            };
        }

        // -------------------- Mesh packing --------------------

        private static (MeshGpu mesh, Dictionary<int, int> meshToBone) BuildMesh(Scene scene, Dictionary<string,int> indexOf, float sceneScale)
        {
            // choose first mesh that has bones
            int mi = -1;
            for (int i = 0; i < scene.MeshCount; i++)
                if (scene.Meshes[i].HasBones) { mi = i; break; }
            if (mi < 0) throw new InvalidOperationException("No skinned mesh found.");

            var m = scene.Meshes[mi];

            // positions/normals/uv
            var vcount = m.VertexCount;
            var verts = new GpuVertex[vcount];
            for (int v = 0; v < vcount; v++)
            {
                verts[v].Position = m.HasVertices ? ToVector3(m.Vertices[v]) * sceneScale : Vector3.Zero;
                verts[v].Normal   = m.HasNormals  ? ToVector3(m.Normals[v])  : Vector3.UnitY;
                if (m.TextureCoordinateChannelCount > 0 && m.HasTextureCoords(0))
                {
                    var t = m.TextureCoordinateChannels[0][v];
                    verts[v].Tex = new Vector2(t.X, t.Y);
                }
            }

            // weights: gather all, then keep top-4 per vertex
            var acc = new List<(int bone, float w)>[vcount];
            for (int v = 0; v < vcount; v++) acc[v] = new List<(int, float)>(8);

            foreach (var b in m.Bones)
            {
                if (!indexOf.TryGetValue(b.Name, out var bi)) continue;
                foreach (var w in b.VertexWeights)
                    acc[w.VertexID].Add((bi, w.Weight));
            }

            for (int v = 0; v < vcount; v++)
            {
                var list = acc[v];
                // sort by weight desc and keep 4
                list.Sort((a, b) => b.w.CompareTo(a.w));
                if (list.Count > 4) list = list.GetRange(0, 4);

                // normalize
                float sum = 0; foreach (var e in list) sum += e.w;
                float inv = sum > 1e-8f ? 1f / sum : 0f;

                // pack
                Vector4 ws = Vector4.Zero;
                Int4 boneIds = new Int4(0);
                for (int i = 0; i < list.Count; i++)
                {
                    var bi = list[i].bone;
                    var ww = list[i].w * inv;
                    if (i == 0) { boneIds.X = bi; ws.X = ww; }
                    else if (i == 1) { boneIds.Y = bi; ws.Y = ww; }
                    else if (i == 2) { boneIds.Z = bi; ws.Z = ww; }
                    else if (i == 3) { boneIds.W = bi; ws.W = ww; }
                }
                verts[v].Bone = boneIds;
                verts[v].Weights = ws;
            }

            // indices
            var indices = new List<int>(m.FaceCount * 3);
            foreach (var f in m.Faces)
            {
                if (f.IndexCount == 3)
                    indices.AddRange(f.Indices);
            }

            var mesh = new MeshGpu { Vertices = verts, Indices = indices.ToArray(), BoneCount = indexOf.Count };
            return (mesh, null!);
        }

        // -------------------- AniMatrixion baking (delta-local DQ) --------------------

        private sealed class TrackDQ
        {
            public int BoneId;
            public float[] Times = Array.Empty<float>(); // ticks
            public DualQuat[] Keys = Array.Empty<DualQuat>(); // LOCAL rigid dq keys
        }

        private static AnimBankGpu BuildAniMatrixions(Scene scene,
                                                   List<string> orderedBones,
                                                   int[] parent,
                                                   Matrix[] bindLocal,
                                                   float fpsIfNeeded, float tpsFallback, float tol,
                                                   float sceneScale)
        {
            var boneIndex = orderedBones.Select((n, i) => (n, i))
                                        .ToDictionary(t => t.n, t => t.i, StringComparer.Ordinal);

            // rest local and its inverse
            var restLocalDQ = bindLocal.Select(MatrixRigidToDQ).ToArray();
            var restInvLocalDQ = restLocalDQ.Select(InverseDQ).ToArray();

            // inverse bind for skin bank
            var bindInvMatrixs = new Matrix[orderedBones.Count];
            // build map name->offset once
            var invBindByName = new Dictionary<string, Matrix>(StringComparer.Ordinal);
            foreach (var mesh in scene.Meshes)
                foreach (var b in mesh.Bones)
                    invBindByName[b.Name] = ToM(b.OffsetMatrix);
            for (int i = 0; i < orderedBones.Count; i++)
                bindInvMatrixs[i] = invBindByName.TryGetValue(orderedBones[i], out var m) ? m : Matrix.Identity;
            var bindInvDQ = bindInvMatrixs.Select(MatrixRigidToDQ).ToArray();

            // collect per-clip tracks
            var clipTracks = new List<List<TrackDQ>>();
            var clipMeta   = new List<(string name, double durTicks, double tps)>();

            for (int c = 0; c < scene.AnimationCount; c++)
            {
                var anim = scene.Animations[c];
                var tps  = anim.TicksPerSecond != 0 ? anim.TicksPerSecond : tpsFallback;
                var durTicks = anim.DurationInTicks;

                var list = new List<TrackDQ>();
                foreach (var ch in anim.NodeAnimationChannels)
                {
                    if (!boneIndex.TryGetValue(ch.NodeName, out var bId)) continue;
                    
                    // Handle cases where either PositionKeys or RotationKeys might be null or empty
                    int posCount = ch.PositionKeys?.Count ?? 0;
                    int rotCount = ch.RotationKeys?.Count ?? 0;
                    int kc = Math.Max(posCount, rotCount);
                    
                    if (kc == 0) continue;  // Skip if no animation data

                    var times = new float[kc];
                    var keys  = new DualQuat[kc];
                    for (int i = 0; i < kc; i++)
                    {
                        var hasP = i < ch.PositionKeys.Count;
                        var hasR = i < ch.RotationKeys.Count;
                        var p = hasP ? ToVector3(ch.PositionKeys[i].Value) * sceneScale : Vector3.Zero;
                        var r = hasR ? ToQ (ch.RotationKeys[i].Value) : Quaternion.Identity;

                        times[i] = hasP ? (float)ch.PositionKeys[i].Time
                                        : (hasR ? (float)ch.RotationKeys[i].Time : 0f);

                        keys[i] = RigidToDQ(r, p); // LOCAL
                    }
                    list.Add(new TrackDQ { BoneId = bId, Times = times, Keys = keys });
                }

                clipTracks.Add(list);
                clipMeta.Add((string.IsNullOrEmpty(anim.Name) ? $"Clip_{c}" : anim.Name, durTicks, tps));
            }

            var localDeltaOut = new List<DualQuat>(4096);
            var clipsOut      = new List<ClipInfoRaster>();

            foreach (var (tracks, meta) in clipTracks.Zip(clipMeta, (t, m) => (t, m)))
            {
                var (_, durTicks, tps) = meta;
                int startLocal = localDeltaOut.Count;

                // detect raster
                int frames; float startTick, stepTick;
                var perTrackTimes = tracks.Select(t => t.Times).ToList();
                if (TryDetectUniformRaster(perTrackTimes, durTicks, out startTick, out stepTick, out frames, tol))
                {
                    EvaluateClipOnRaster(tracks, orderedBones.Count, parent,
                                         restInvLocalDQ, bindInvDQ,
                                         frames, startTick, stepTick,
                                         tt => tt,
                                         localDeltaOut);
                }
                else
                {
                    var step = (float)(tps / Math.Max(1e-6, fpsIfNeeded));
                    frames = Math.Max(1, (int)Math.Ceiling(durTicks / step) + 1);
                    startTick = 0f; stepTick = step;

                    EvaluateClipOnRaster(tracks, orderedBones.Count, parent,
                                         restInvLocalDQ, bindInvDQ,
                                         frames, startTick, stepTick,
                                         tt => tt,
                                         localDeltaOut);
                }

                clipsOut.Add(new ClipInfoRaster {
                    BoneCount = orderedBones.Count,
                    FrameCount = frames,
                    StartIndex = startLocal,
                    TicksPerSecond = (float)tps,
                    StartTick = startTick,
                    StepTick = stepTick,
                    DurationSeconds = (float)(tps > 0 ? durTicks / tps : 0)
                });
            }

            return new AnimBankGpu {
                LocalDeltaDq = localDeltaOut.ToArray(),
                Clips = clipsOut.ToArray()
            };
        }

        private static void EvaluateClipOnRaster(
            List<TrackDQ> tracks,
            int boneCount,
            int[] parent,
            DualQuat[] restInvLocalDQ,
            DualQuat[] bindInvDQ,
            int frameCount,
            float startTick,
            float stepTick,
            Func<float,float> ticksFn,
            List<DualQuat> appendLocalDelta)
        {
            var trByBone = new TrackDQ[boneCount];
            foreach (var tr in tracks) trByBone[tr.BoneId] = tr;

            var local  = new DualQuat[boneCount];
            var global = new DualQuat[boneCount];

            for (int f = 0; f < frameCount; f++)
            {
                float tt = ticksFn(startTick + f * stepTick);

                // sample local
                for (int b = 0; b < boneCount; b++)
                {
                    var tr = trByBone[b];
                    if (tr == null || tr.Keys.Length == 0) { local[b] = IdentityDQ(); continue; }

                    int i0 = FindKey(tr.Times, tt);
                    int i1 = Math.Min(i0 + 1, tr.Keys.Length - 1);
                    float t0 = tr.Times[i0], t1 = tr.Times[i1];
                    float u = (t1 > t0) ? (tt - t0) / (t1 - t0) : 0f;

                    var dq0 = tr.Keys[i0];
                    var dq1 = tr.Keys[i1];
                    var qr  = Slerp(dq0.Qr, dq1.Qr, u);
                    var qd  = Vector4.Lerp(dq0.Qd, dq1.Qd, u);
                    OrthoNormalize(ref qr, ref qd);
                    local[b] = new DualQuat { Qr = qr, Qd = qd };
                }

                // delta-local = restInvLocal * currentLocal
                for (int b = 0; b < boneCount; b++)
                {
                    var ri = restInvLocalDQ[b];
                    var cl = local[b];
                    var sr = QMul(ri.Qr, cl.Qr);
                    var sd = QMul(ri.Qr, cl.Qd) + QMul(ri.Qd, cl.Qr);
                    OrthoNormalize(ref sr, ref sd);
                    appendLocalDelta.Add(new DualQuat { Qr = sr, Qd = sd });
                }

               
            }
        }
        
        

        private static HashSet<string> GatherBoneNames(Scene scene)
        {
            var set = new HashSet<string>(StringComparer.Ordinal);
            foreach (var mesh in scene.Meshes)
                foreach (var b in mesh.Bones)
                    set.Add(b.Name);
            if (set.Count == 0)
                throw new InvalidOperationException("No bones found.");
            return set;
        }

        // -------------------- Math helpers (DQ) --------------------

        private static DualQuat IdentityDQ() => new DualQuat { Qr = new Vector4(0,0,0,1), Qd = Vector4.Zero };

        private static Vector4 QMul(in Vector4 a, in Vector4 b) => new Vector4(
            a.W*b.X + a.X*b.W + a.Y*b.Z - a.Z*b.Y,
            a.W*b.Y - a.X*b.Z + a.Y*b.W + a.Z*b.X,
            a.W*b.Z + a.X*b.Y - a.Y*b.X + a.Z*b.W,
            a.W*b.W - (a.X*b.X + a.Y*b.Y + a.Z*b.Z)
        );

        private static Vector4 Slerp(Vector4 a, Vector4 b, float t)
        {
            float d = Vector4.Dot(a, b);
            if (d < 0) b = -b;
            float theta = MathF.Acos(Math.Clamp(Vector4.Dot(a, b), -1f, 1f));
            if (theta < 1e-5f) return Vector4.Normalize(Vector4.Lerp(a, b, t));
            float s = MathF.Sin(theta);
            var r = (MathF.Sin((1 - t) * theta) / s) * a + (MathF.Sin(t * theta) / s) * b;
            return Vector4.Normalize(r);
        }

        private static void OrthoNormalize(ref Vector4 qr, ref Vector4 qd)
        {
            qr = Vector4.Normalize(qr);
            qd = qd - Vector4.Dot(qr, qd) * qr;
        }

        private static DualQuat InverseDQ(DualQuat q)
        {
            var qc = new Vector4(-q.Qr.X, -q.Qr.Y, -q.Qr.Z, q.Qr.W);   // conj
            var rd = -QMul(QMul(qc, q.Qd), qc);
            return new DualQuat { Qr = qc, Qd = rd };
        }

        private static DualQuat RigidToDQ(Quaternion r, Vector3 t)
        {
            var qr = Vector4.Normalize(new Vector4(r.X, r.Y, r.Z, r.W));
            var tq = new Vector4(t, 0f);
            var qd = 0.5f * QMul(tq, qr);
            return new DualQuat { Qr = qr, Qd = qd };
        }

        private static DualQuat MatrixRigidToDQ(Matrix m)
        {
            // Extract just the 3x3 rotation part
            var r = new Quaternion();
            r = Quaternion.RotationMatrix(m);
            
            // Get the translation part
            var t = new Vector3(m.M41, m.M42, m.M43);
            
            return RigidToDQ(r, t);
        }

        // -------------------- Util --------------------

        private static int FindKey(float[] times, float t)
        {
            if (times == null || times.Length <= 1) return 0;
            int lo = 0, hi = times.Length - 1;
            while (hi - lo > 1)
            {
                int mid = (lo + hi) >> 1;
                if (times[mid] <= t) lo = mid; else hi = mid;
            }
            return lo;
        }

        private static Vector3 ToVector3(Vector3D v) => new(v.X, v.Y, v.Z);
        private static Quaternion ToQ(Assimp.Quaternion q) => new(q.X, q.Y, q.Z, q.W);

        private static Matrix ToM(Matrix4x4 m) =>
            new Matrix(m.A1,m.B1,m.C1,m.D1, m.A2,m.B2,m.C2,m.D2, m.A3,m.B3,m.C3,m.D3, m.A4,m.B4,m.C4,m.D4);

        private static global::Assimp.Matrix4x4 FromM(Matrix m) =>
            new global::Assimp.Matrix4x4(
                m.M11, m.M21, m.M31, m.M41,
                m.M12, m.M22, m.M32, m.M42,
                m.M13, m.M23, m.M33, m.M43,
                m.M14, m.M24, m.M34, m.M44);

        private static bool TryDetectUniformRaster(IReadOnlyList<float[]> perTrackTimes, double clipDurTicks,
                                                   out float start, out float step, out int frames,
                                                   float tol)
        {
            start = 0; step = 0; frames = 1;
            float[]? refT = null;
            foreach (var t in perTrackTimes) { if (t != null && t.Length >= 2) { refT = t; break; } }
            if (refT == null) { start = 0f; step = (float)clipDurTicks; frames = 1; return true; }

            float s = refT[0];
            float stp = (refT[^1] - refT[0]) / (refT.Length - 1);
            if (stp <= 0) return false;

            for (int i = 1; i < refT.Length; i++)
            {
                float ideal = s + i * stp;
                if (MathF.Abs(refT[i] - ideal) > tol) return false;
            }

            foreach (var t in perTrackTimes)
            {
                if (t == null || t.Length <= 1) continue;
                if (t.Length == refT.Length)
                {
                    float ts = t[0];
                    float tt = (t[^1] - t[0]) / (t.Length - 1);
                    if (MathF.Abs(ts - s) > tol) return false;
                    if (MathF.Abs(tt - stp) > tol) return false;
                    for (int i = 1; i < t.Length; i++)
                    {
                        float ideal = ts + i * tt;
                        if (MathF.Abs(t[i] - ideal) > tol) return false;
                    }
                }
                else
                {
                    for (int i = 0; i < t.Length; i++)
                    {
                        float k = (t[i] - s) / stp;
                        if (MathF.Abs(k - MathF.Round(k)) > 1e-3f) return false;
                    }
                }
            }

            frames = Math.Max(1, (int)Math.Round((clipDurTicks - s) / stp + 1));
            start = s; step = stp;
            return true;
        }

        private static void Walk(Node n, Action<Node> fn)
        {
            fn(n);
            foreach (var c in n.Children) Walk(c, fn);
        }
    }

    // -------------------- Packing helpers for GPU buffers --------------------

    public static class GpuPacking
    {
        /// Pack DualQuats into float4[] layout [ qr0, qd0, qr1, qd1, ... ] for a clip.
        /// Base = clip.StartIndex; Count per frame = 2 * BoneCount float4s.
        public static Vector4[] PackDQ(DualQuat[] dqs)
        {
            var out4 = new Vector4[dqs.Length * 2];
            for (int i = 0; i < dqs.Length; i++)
            {
                out4[2*i+0] = dqs[i].Qr;
                out4[2*i+1] = dqs[i].Qd;
            }
            return out4;
        }

        /// Convert a frame selection (frame, boneIndex) to the packed float4 index.
        public static int DQIndex(int start, int boneCount, int frame, int bone) => (start + frame * boneCount + bone) * 2;
    }
}
