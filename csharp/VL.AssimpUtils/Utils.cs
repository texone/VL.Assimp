using Assimp;
using Quaternion = Stride.Core.Mathematics.Quaternion;

#pragma warning disable CS1591 // Missing XML comment for publicly visible type or member

namespace VL.AssimpGpu;

// -------------------- GPU structs --------------------

/// <summary>
/// Vertex layout for GPU (StructuredBuffer); tightly packed / sequential.
/// </summary>
public struct GpuVertex
{
    public Vector3 Position;     // 12
    public Vector3 Normal;       // 24
    public Vector2 Tex;          // 32
    public Int4 Bone;            // 48
    public Vector4 Weights;      // 64 bytes total
}

public struct SkeletonGpu
{
    public int[] Parent;        // bone -> parent (-1 root)
    public Matrix[] InverseBind;   // bind^-1 (Assimp Bone.OffsetMatrix)
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
    private struct SkeletonBuildResult
    {
        public List<string> OrderedBoneNames;
        public Dictionary<string, int> IndexOf;
        public int[] ParentIndices;
        public Dictionary<string, Node> NodeByName;
    }

    private static SkeletonBuildResult BuildRobustSkeleton(Scene scene)
    {
        var nodeByName = new Dictionary<string, Node>(StringComparer.Ordinal);
        Walk(scene.RootNode, n => nodeByName[n.Name] = n);

        var weightedBoneNames = new HashSet<string>(StringComparer.Ordinal);
        foreach (var mesh in scene.Meshes)
            foreach (var bone in mesh.Bones)
                weightedBoneNames.Add(bone.Name);

        if (weightedBoneNames.Count == 0)
            throw new InvalidOperationException("No bones with vertex weights found in the scene.");

        static List<Node> GetPathToRoot(Node node)
        {
            var path = new List<Node>();
            while (node != null) { path.Add(node); node = node.Parent; }
            path.Reverse();
            return path;
        }

        var bonePaths = weightedBoneNames
            .Where(name => nodeByName.ContainsKey(name))
            .Select(name => GetPathToRoot(nodeByName[name]))
            .ToList();

        if (bonePaths.Count == 0)
            throw new InvalidOperationException("Could not find any of the weighted bones in the scene graph.");

        var rootPath = new List<Node>(bonePaths[0]);
        foreach (var path in bonePaths.Skip(1))
        {
            int commonDepth = 0;
            for (int i = 0; i < Math.Min(rootPath.Count, path.Count); i++)
            {
                if (rootPath[i] != path[i]) break;
                commonDepth = i + 1;
            }
            rootPath.RemoveRange(commonDepth, rootPath.Count - commonDepth);
        }

        var skeletonRootNode = rootPath.LastOrDefault();
        if (skeletonRootNode == null)
            throw new InvalidOperationException("Failed to find a common skeleton root.");

        var orderedBoneNames = new List<string>();
        Walk(skeletonRootNode, n => orderedBoneNames.Add(n.Name));

        var indexOf = orderedBoneNames.Select((name, i) => (name, i))
                                      .ToDictionary(t => t.name, t => t.i, StringComparer.Ordinal);

        var parentIndices = new int[orderedBoneNames.Count];
        for (int i = 0; i < orderedBoneNames.Count; i++)
        {
            var name = orderedBoneNames[i];
            var node = nodeByName[name];

            if (node == skeletonRootNode || node.Parent == null || !indexOf.TryGetValue(node.Parent.Name, out var parentIndex))
            {
                parentIndices[i] = -1;
            }
            else
            {
                parentIndices[i] = parentIndex;
            }
        }

        return new SkeletonBuildResult
        {
            OrderedBoneNames = orderedBoneNames,
            IndexOf = indexOf,
            ParentIndices = parentIndices,
            NodeByName = nodeByName
        };
    }

    public static ModelGpu LoadFbx(string path, float sceneScale = 0.01f, float fpsIfNeeded = 30f)
    {
        try
        {
            var ctx = new AssimpContext();
            var pps = PostProcessSteps.Triangulate
                    | PostProcessSteps.JoinIdenticalVertices
                    | PostProcessSteps.LimitBoneWeights
                    | PostProcessSteps.ImproveCacheLocality
                    | PostProcessSteps.OptimizeMeshes
                    | PostProcessSteps.OptimizeGraph;

            var scene = ctx.ImportFile(path, pps);
            if (scene == null || scene.RootNode == null)
                throw new InvalidOperationException("Failed to load scene.");

            var skeletonBuild = BuildRobustSkeleton(scene);
            var orderedBones = skeletonBuild.OrderedBoneNames;
            var indexOf = skeletonBuild.IndexOf;
            var nodeByName = skeletonBuild.NodeByName;
            int nb = orderedBones.Count;

            // Load original, unscaled local transforms for the bind pose
            var bindLocal = new Matrix[nb];
            for (int i = 0; i < nb; i++)
            {
                var name = orderedBones[i];
                bindLocal[i] = nodeByName.TryGetValue(name, out var nn) ? ToM(nn.Transform) : Matrix.Identity;
            }

            // Load original InverseBind matrices from the file
            var invBindByName = new Dictionary<string, Matrix>(StringComparer.Ordinal);
            foreach (var mesh in scene.Meshes)
                foreach (var bone in mesh.Bones)
                    invBindByName[bone.Name] = ToM(bone.OffsetMatrix);

            var invBind = new Matrix[nb];
            var scaleMatrix = Matrix.Scaling(sceneScale);
            var inverseScaleMatrix = Matrix.Scaling(1.0f / sceneScale);

            for (int i = 0; i < nb; i++)
            {
                var name = orderedBones[i];
                var originalInvBind = invBindByName.TryGetValue(name, out var m) ? m : Matrix.Identity;

                invBind[i] = inverseScaleMatrix * originalInvBind * scaleMatrix;
            }

            var skeleton = new SkeletonGpu
            {
                Parent = skeletonBuild.ParentIndices,
                InverseBind = invBind,
                BindLocal = bindLocal
            };

            var meshGpu = BuildMesh(scene, indexOf, sceneScale);
            var anim = BuildAnimations(scene, orderedBones, bindLocal, fpsIfNeeded, sceneScale);

            return new ModelGpu
            {
                Skeleton = skeleton,
                Mesh = meshGpu,
                Anim = anim,
                BoneNames = orderedBones.ToArray()
            };
        }
        catch (Exception e)
        {

            System.Diagnostics.Debug.WriteLine($"Failed to load model '{path}'. Reason: {e.Message}. Returning default quad.");
            return BuildDefault();
        }
    }

    private static MeshGpu BuildMesh(Scene scene, Dictionary<string, int> indexOf, float sceneScale)
    {
        var relevantMeshes = scene.Meshes.Where(m => m.HasBones && m.VertexCount > 0).ToList();
        if (relevantMeshes.Count == 0)
            throw new InvalidOperationException("No skinned mesh with vertices found.");

        var totalVertices = relevantMeshes.Sum(m => m.VertexCount);
        var totalIndices = relevantMeshes.Sum(m => m.FaceCount * 3);

        var vertices = new GpuVertex[totalVertices];
        var indices = new int[totalIndices];
        var vertexWeights = new List<(int bone, float w)>[totalVertices];
        for (int i = 0; i < totalVertices; i++) vertexWeights[i] = new List<(int, float)>();

        int vertexOffset = 0;
        int indexOffset = 0;

        foreach (var m in relevantMeshes)
        {
            for (int v = 0; v < m.VertexCount; v++)
            {
                vertices[vertexOffset + v].Position = m.HasVertices ? ToVector3(m.Vertices[v]) * sceneScale : Vector3.Zero;
                vertices[vertexOffset + v].Normal = m.HasNormals ? ToVector3(m.Normals[v]) : Vector3.UnitY;
                if (m.HasTextureCoords(0))
                {
                    var t = m.TextureCoordinateChannels[0][v];
                    vertices[vertexOffset + v].Tex = new Vector2(t.X, t.Y);
                }
            }

            foreach (var b in m.Bones)
            {
                if (!indexOf.TryGetValue(b.Name, out var bi)) continue;
                foreach (var w in b.VertexWeights)
                {
                    if (w.VertexID < m.VertexCount)
                        vertexWeights[vertexOffset + w.VertexID].Add((bi, w.Weight));
                }
            }

            foreach (var f in m.Faces)
            {
                if (f.IndexCount == 3)
                {
                    indices[indexOffset++] = vertexOffset + f.Indices[0];
                    indices[indexOffset++] = vertexOffset + f.Indices[1];
                    indices[indexOffset++] = vertexOffset + f.Indices[2];
                }
            }
            vertexOffset += m.VertexCount;
        }

        for (int v = 0; v < totalVertices; v++)
        {
            var list = vertexWeights[v].OrderByDescending(w => w.w).Take(4).ToList();
            float sum = list.Sum(w => w.w);
            float inv = sum > 1e-8f ? 1f / sum : 0f;

            var boneIds = new Int4(0);
            var ws = Vector4.Zero;
            if (list.Count > 0) { boneIds.X = list[0].bone; ws.X = list[0].w * inv; }
            if (list.Count > 1) { boneIds.Y = list[1].bone; ws.Y = list[1].w * inv; }
            if (list.Count > 2) { boneIds.Z = list[2].bone; ws.Z = list[2].w * inv; }
            if (list.Count > 3) { boneIds.W = list[3].bone; ws.W = list[3].w * inv; }
            vertices[v].Bone = boneIds;
            vertices[v].Weights = ws;
        }

        return new MeshGpu 
        { 
            Vertices = vertices, 
            Indices = indices, 
            BoneCount = indexOf.Count 
        };
    }

    private static AnimBankGpu BuildAnimations(Scene scene,
                                               List<string> orderedBones,
                                               Matrix[] bindLocal,
                                               float fpsIfNeeded,
                                               float sceneScale)
    {
        var boneIndex = orderedBones.Select((n, i) => (n, i)).ToDictionary(t => t.n, t => t.i, StringComparer.Ordinal);

        var bindPoseDecomposed = new (Vector3 S, Quaternion R, Vector3 T)[bindLocal.Length];
        for (int i = 0; i < bindLocal.Length; i++)
        {
            bindLocal[i].Decompose(out bindPoseDecomposed[i].S, out bindPoseDecomposed[i].R, out bindPoseDecomposed[i].T);
        }

        var finalDqBuffer = new List<DualQuat>();
        var finalClips = new List<ClipInfoRaster>();

        foreach (var anim in scene.Animations)
        {
            var tps = anim.TicksPerSecond > 0 ? anim.TicksPerSecond : 25.0;
            var durTicks = anim.DurationInTicks;

            var channelsByBoneId = anim.NodeAnimationChannels
                .Where(ch => boneIndex.ContainsKey(ch.NodeName))
                .ToDictionary(ch => boneIndex[ch.NodeName]);

            var step = (float)(tps / Math.Max(1e-6, fpsIfNeeded));
            var frameCount = Math.Max(1, (int)Math.Ceiling(durTicks / step) + 1);
            int clipStartIndex = finalDqBuffer.Count;

            for (int f = 0; f < frameCount; f++)
            {
                double tick = f * step;
                for (int b = 0; b < orderedBones.Count; b++)
                {
                    var finalS = bindPoseDecomposed[b].S;
                    var finalR = bindPoseDecomposed[b].R;
                    var finalT = bindPoseDecomposed[b].T;

                    if (channelsByBoneId.TryGetValue(b, out var channel))
                    {
                        if (channel.HasPositionKeys)
                        {
                            int i0 = FindPositionKey(channel.PositionKeys, tick);
                            int i1 = Math.Min(i0 + 1, channel.PositionKeys.Count - 1);
                            var k0 = channel.PositionKeys[i0]; var k1 = channel.PositionKeys[i1];
                            double timeDiff = k1.Time - k0.Time;
                            float u = (timeDiff > 0) ? (float)((tick - k0.Time) / timeDiff) : 0f;
                            var interpolatedPos = Vector3.Lerp(ToVector3(k0.Value), ToVector3(k1.Value), u);
                            finalT = interpolatedPos;
                        }

                        if (channel.HasRotationKeys)
                        {
                            int i0 = FindRotationKey(channel.RotationKeys, tick);
                            int i1 = Math.Min(i0 + 1, channel.RotationKeys.Count - 1);
                            var k0 = channel.RotationKeys[i0]; var k1 = channel.RotationKeys[i1];
                            double timeDiff = k1.Time - k0.Time;
                            float u = (timeDiff > 0) ? (float)((tick - k0.Time) / timeDiff) : 0f;
                            finalR = Quaternion.Slerp(ToQ(k0.Value), ToQ(k1.Value), u);
                        }
                    }

                    var finalLocalMatrix = Matrix.Scaling(finalS) *
                                           Matrix.RotationQuaternion(finalR) *
                                           Matrix.Translation(finalT * sceneScale);

                    finalDqBuffer.Add(MatrixRigidToDQ(finalLocalMatrix));
                }
            }

            finalClips.Add(new ClipInfoRaster
            {
                BoneCount = orderedBones.Count,
                FrameCount = frameCount,
                StartIndex = clipStartIndex,
                TicksPerSecond = (float)tps,
                StartTick = 0,
                StepTick = step,
                DurationSeconds = (float)(tps > 0 ? durTicks / tps : 0)
            });
        }

        return new AnimBankGpu
        {
            LocalDeltaDq = finalDqBuffer.ToArray(),
            Clips = finalClips.ToArray()
        };
    }

    private static int FindPositionKey(IReadOnlyList<VectorKey> keys, double time)
    {
        if (keys.Count <= 1) return 0;
        int lo = 0, hi = keys.Count - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) >> 1;
            if (keys[mid].Time <= time) lo = mid; else hi = mid;
        }
        return lo;
    }

    private static int FindRotationKey(IReadOnlyList<QuaternionKey> keys, double time)
    {
        if (keys.Count <= 1) return 0;
        int lo = 0, hi = keys.Count - 1;
        while (hi - lo > 1)
        {
            int mid = (lo + hi) >> 1;
            if (keys[mid].Time <= time) lo = mid; else hi = mid;
        }
        return lo;
    }

    /// <summary>
    /// Creates a minimal, 1x1 unit-sized, valid ModelGpu object to prevent downstream failures.
    /// The quad has correct winding for a right-handed system (Y-up, front-face pointing to +Z).
    /// </summary>
    private static ModelGpu BuildDefault()
    {
        // 1. A simple 1x1 quad mesh bound to one bone
        var vertices = new GpuVertex[4];
        
        // Positions are centered, making a 1x1 quad. Normal points towards positive Z.
        // Tex Coords have (0,0) at the top-left.
        vertices[0] = new GpuVertex { Position = new Vector3(-0.5f, -0.5f, 0), Normal = Vector3.UnitZ, Tex = new Vector2(0, 1), Bone = new Int4(0), Weights = Vector4.UnitX }; // Bottom-left
        vertices[1] = new GpuVertex { Position = new Vector3(0.5f, -0.5f, 0), Normal = Vector3.UnitZ, Tex = new Vector2(1, 1), Bone = new Int4(0), Weights = Vector4.UnitX }; // Bottom-right
        vertices[2] = new GpuVertex { Position = new Vector3(0.5f, 0.5f, 0), Normal = Vector3.UnitZ, Tex = new Vector2(1, 0), Bone = new Int4(0), Weights = Vector4.UnitX }; // Top-right
        vertices[3] = new GpuVertex { Position = new Vector3(-0.5f, 0.5f, 0), Normal = Vector3.UnitZ, Tex = new Vector2(0, 0), Bone = new Int4(0), Weights = Vector4.UnitX }; // Top-left

        var mesh = new MeshGpu
        {
            Vertices = vertices,
            // Indices are in Counter-Clockwise (CCW) order for right-handed rendering
            Indices = [0, 1, 2, 0, 2, 3],
            BoneCount = 1
        };

        // 2. A minimal skeleton with a single root bone
        var skeleton = new SkeletonGpu
        {
            Parent = [-1],
            InverseBind = [Matrix.Identity],
            BindLocal = [Matrix.Identity]
        };

        // 3. A minimal animation bank with one clip containing one identity frame
        var anim = new AnimBankGpu
        {
            LocalDeltaDq = [IdentityDQ()],
            Clips =
            [
                new ClipInfoRaster
                {
                    BoneCount = 1,
                    FrameCount = 1,
                    StartIndex = 0,
                    TicksPerSecond = 30,
                    StartTick = 0,
                    StepTick = 1,
                    DurationSeconds = 0
                }
            ]
        };

        return new ModelGpu
        {
            Skeleton = skeleton,
            Mesh = mesh,
            Anim = anim,
            BoneNames = ["DefaultRoot"]
        };
    }

    private static DualQuat IdentityDQ() => new DualQuat { Qr = new Vector4(0, 0, 0, 1), Qd = Vector4.Zero };

    private static Vector4 QMul(in Vector4 a, in Vector4 b) =>
        new Vector4(a.W * b.X + a.X * b.W + a.Y * b.Z - a.Z * b.Y,
                    a.W * b.Y - a.X * b.Z + a.Y * b.W + a.Z * b.X,
                    a.W * b.Z + a.X * b.Y - a.Y * b.X + a.Z * b.W,
                    a.W * b.W - (a.X * b.X + a.Y * b.Y + a.Z * b.Z));

    private static DualQuat MatrixRigidToDQ(Matrix m)
    {
        m.Decompose(out _, out Quaternion r, out Vector3 t);
        return RigidToDQ(r, t);
    }

    private static DualQuat RigidToDQ(Quaternion r, Vector3 t)
    {
        var qr = new Vector4(r.X, r.Y, r.Z, -r.W); // finding this minus sign did cost about 3 days of debugging
        var tq = new Vector4(t, 0f);
        var qd = 0.5f * QMul(tq, qr);
        return new DualQuat { Qr = qr, Qd = qd };
    }

    private static Vector3 ToVector3(Assimp.Vector3D v) => new Vector3(v.X, v.Y, v.Z);

    private static Quaternion ToQ(Assimp.Quaternion q) => new Quaternion(q.X, q.Y, q.Z, q.W);

    private static Matrix ToM(Assimp.Matrix4x4 m) =>
        new Matrix(m.A1, m.B1, m.C1, m.D1,
                   m.A2, m.B2, m.C2, m.D2,
                   m.A3, m.B3, m.C3, m.D3,
                   m.A4, m.B4, m.C4, m.D4);

    private static void Walk(Node n, Action<Node> fn)
    {
        fn(n);
        foreach (var c in n.Children)
        {
            Walk(c, fn);
        }
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
            out4[2 * i + 0] = dqs[i].Qr;
            out4[2 * i + 1] = dqs[i].Qd;
        }
        return out4;
    }

    /// Convert a frame selection (frame, boneIndex) to the packed float4 index.
    public static int DQIndex(int start, int boneCount, int frame, int bone) => (start + frame * boneCount + bone) * 2;
}