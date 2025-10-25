// For examples, see:
// https://thegraybook.vvvv.org/reference/extending/writing-nodes.html#examples

using System;
using System.Collections.Generic;
using System.Linq;
using Assimp;
using Matrix4x4 = System.Numerics.Matrix4x4;
using Quaternion = System.Numerics.Quaternion;
using Vector3 = System.Numerics.Vector3;
using Vector4 = System.Numerics.Vector4;

namespace VL.Assimp;

public struct DualQuat { public Vector4 qr, qd; }

public sealed class DqAnimBankRastered
{
    // Final, skin-ready dual quaternions (global * bindInv), all clips concatenated
    public DualQuat[] SkinDQ = Array.Empty<DualQuat>();

    // Skeleton
    public string[] BoneNames = Array.Empty<string>();
    public int[] ParentIndex = Array.Empty<int>();     // bone -> parent (-1 root)

    // Clip table
    public ClipInfoRaster[] Clips = Array.Empty<ClipInfoRaster>();
}

public struct ClipInfoRaster
{
    public int BoneCount;
    public int FrameCount;
    public int StartIndex;          // index into SkinDQ (DualQuat units)
    public float TicksPerSecond;   // tps used for this clip
    public float StartTick;         // usually 0
    public float StepTick;          // ticks per frame (uniform)
    public float DurationSeconds;
}

public static class DqRasteredBuilder
{
    public static DqAnimBankRastered BuildRastered(Scene scene, float fpsIfNeeded = 30f, float tpsFallback = 25f, float uniformTolerance = 1e-4f)
    {
        if (scene == null) throw new ArgumentNullException(nameof(scene));
        if (!scene.HasMeshes) throw new InvalidOperationException("Scene has no meshes.");

        // --- 1) Bone set (stable order), parents, bind inverse ---
        var boneSet = new HashSet<string>(StringComparer.Ordinal);
        foreach (var m in scene.Meshes)
            foreach (var b in m.Bones)
                boneSet.Add(b.Name);

        var boneNames = new List<string>();
        Walk(scene.RootNode, n => { if (boneSet.Contains(n.Name)) boneNames.Add(n.Name); });
        foreach (var n in boneSet) if (!boneNames.Contains(n)) boneNames.Add(n);

        var boneIndex = boneNames.Select((n, i) => (n, i))
                                 .ToDictionary(t => t.n, t => t.i, StringComparer.Ordinal);

        var parentIndex = new int[boneNames.Count]; Array.Fill(parentIndex, -1);
        var parentByName = new Dictionary<string, string>(StringComparer.Ordinal);
        Walk(scene.RootNode, n => { foreach (var c in n.Children) parentByName[c.Name] = n.Name; });
        foreach (var kv in boneIndex)
            if (parentByName.TryGetValue(kv.Key, out var pn) && boneIndex.TryGetValue(pn, out var pid))
                parentIndex[kv.Value] = pid;

        // offset = inverse bind in Assimp
        var bindInvMats = Enumerable.Repeat(Matrix4x4.Identity, boneNames.Count).ToArray();
        foreach (var mesh in scene.Meshes)
            foreach (var b in mesh.Bones)
                if (boneIndex.TryGetValue(b.Name, out var bi)) bindInvMats[bi] = ToM(b.OffsetMatrix);
        var bindInvDQ = bindInvMats.Select(MatRigidToDQ).ToArray();

        // --- 2) Collect per-clip track data (key times + local DQ keys) ---
        // Map clip → list of (boneId, keyTimes[], keyDQ[])
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

                // Build key-aligned local rigid DQs per key index (use whichever keys are present)
                int kc = Math.Max(ch.PositionKeys.Count, ch.RotationKeys.Count);
                if (kc == 0) continue;

                var times = new float[kc];
                var dq    = new DualQuat[kc];

                for (int i = 0; i < kc; i++)
                {
                    var hasP = i < ch.PositionKeys.Count;
                    var hasR = i < ch.RotationKeys.Count;
                    var p = hasP ? ToV3(ch.PositionKeys[i].Value) : Vector3.Zero;
                    var r = hasR ? ToQ (ch.RotationKeys[i].Value) : Quaternion.Identity;

                    times[i] = hasP ? (float)ch.PositionKeys[i].Time
                                    : (hasR ? (float)ch.RotationKeys[i].Time : 0f);

                    dq[i] = RigidToDQ(r, p);
                }

                list.Add(new TrackDQ { BoneId = bId, Times = times, Keys = dq });
            }

            clipTracks.Add(list);
            clipMeta.Add((string.IsNullOrEmpty(anim.Name) ? $"Clip_{c}" : anim.Name, durTicks, tps));
        }

        // --- 3) For each clip, detect/reuse uniform raster or resample to FPS ---
        var allSkin = new List<DualQuat>(1024);
        var clipsOut = new List<ClipInfoRaster>();

        foreach (var (tracks, meta) in clipTracks.Zip(clipMeta, (t, m) => (t, m)))
        {
            var (name, durTicks, tps) = meta;
            if (tracks.Count == 0)
            {
                // No tracks → 1-frame identity raster
                int start = allSkin.Count;
                var oneFrame = Enumerable.Repeat(IdentityDQ(), boneNames.Count).ToArray();
                allSkin.AddRange(oneFrame);

                clipsOut.Add(new ClipInfoRaster {
                    BoneCount = boneNames.Count, FrameCount = 1, StartIndex = start,
                    TicksPerSecond = (float)tps, StartTick = 0, StepTick = (float)durTicks,
                    DurationSeconds = (float)(tps > 0 ? durTicks / tps : 0)
                });
                continue;
            }

            // Gather per-track times for raster detection
            var perTrackTimes = tracks.Select(tr => tr.Times).ToList();
            if (TryDetectUniformRaster(perTrackTimes, durTicks, out var startTick, out var stepTick, out var frameCount, uniformTolerance))
            {
                // Reuse raster: sample at each frame time, build skin palette
                int start = allSkin.Count;
                EvaluateClipOnRaster(tracks, boneNames.Count, parentIndex, bindInvDQ,
                                     frameCount, startTick, stepTick,
                                     (tt) => tt, // times already in ticks
                                     allSkin);

                clipsOut.Add(new ClipInfoRaster {
                    BoneCount = boneNames.Count, FrameCount = frameCount, StartIndex = start,
                    TicksPerSecond = (float)tps, StartTick = startTick, StepTick = stepTick,
                    DurationSeconds = (float)(tps > 0 ? durTicks / tps : 0)
                });
            }
            else
            {
                // Create uniform raster @ fpsIfNeeded
                var step = (float)(tps / Math.Max(1e-6, fpsIfNeeded));
                var frames = Math.Max(1, (int)Math.Ceiling(durTicks / step) + 1);

                int start = allSkin.Count;
                EvaluateClipOnRaster(tracks, boneNames.Count, parentIndex, bindInvDQ,
                                     frames, 0f, step,
                                     (tt) => tt, // ticks
                                     allSkin);

                clipsOut.Add(new ClipInfoRaster {
                    BoneCount = boneNames.Count, FrameCount = frames, StartIndex = start,
                    TicksPerSecond = (float)tps, StartTick = 0f, StepTick = step,
                    DurationSeconds = (float)(tps > 0 ? durTicks / tps : 0)
                });
            }
        }

        return new DqAnimBankRastered {
            SkinDQ = allSkin.ToArray(),
            BoneNames = boneNames.ToArray(),
            ParentIndex = parentIndex,
            Clips = clipsOut.ToArray()
        };
    }

    // ---------- Internal types ----------

    private sealed class TrackDQ
    {
        public int BoneId;
        public float[] Times = Array.Empty<float>();   // ticks
        public DualQuat[] Keys = Array.Empty<DualQuat>(); // local rigid DQ keys
    }

    // ---------- Raster detection ----------

    private static bool TryDetectUniformRaster(IReadOnlyList<float[]> perTrackTimes, double clipDurTicks,
                                               out float start, out float step, out int frames,
                                               float tol)
    {
        start = 0; step = 0; frames = 1;

        // find a reference track with >= 2 keys
        float[]? refT = null;
        foreach (var t in perTrackTimes) { if (t != null && t.Length >= 2) { refT = t; break; } }
        if (refT == null) { start = 0f; step = (float)clipDurTicks; frames = 1; return true; }

        float s = refT[0];
        float stp = (refT[^1] - refT[0]) / (refT.Length - 1);
        if (stp <= 0) return false;

        // verify reference uniform
        for (int i = 1; i < refT.Length; i++)
        {
            float ideal = s + i * stp;
            if (MathF.Abs(refT[i] - ideal) > tol) return false;
        }

        // verify all tracks align to the same grid (they may have fewer keys)
        foreach (var t in perTrackTimes)
        {
            if (t == null || t.Length <= 1) continue;
            if (t.Length == refT.Length)
            {
                // same count => same start/step and uniformity
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
                // different count: every key must lie on the ref grid
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

    // ---------- Evaluate on raster (build SkinDQ) ----------

    private static void EvaluateClipOnRaster(
        List<TrackDQ> tracks,
        int boneCount,
        int[] parentIndex,
        DualQuat[] bindInvDQ,
        int frameCount,
        float startTick,
        float stepTick,
        Func<float, float> ticksFn,
        List<DualQuat> appendTo)
    {
        // map bone -> track (assume <= 1 track per bone for this clip)
        var trByBone = new TrackDQ[boneCount];
        for (int i = 0; i < boneCount; i++) trByBone[i] = null!;
        foreach (var tr in tracks) trByBone[tr.BoneId] = tr;

        var local = new DualQuat[boneCount];
        var global = new DualQuat[boneCount];

        for (int f = 0; f < frameCount; f++)
        {
            float tt = startTick + f * stepTick;
            tt = ticksFn(tt);

            // 1) sample local
            for (int b = 0; b < boneCount; b++)
            {
                var tr = trByBone[b];
                if (tr == null || tr.Keys.Length == 0)
                {
                    local[b] = IdentityDQ();
                    continue;
                }

                int i0 = FindKey(tr.Times, tt);
                int i1 = Math.Min(i0 + 1, tr.Keys.Length - 1);
                float t0 = tr.Times[i0];
                float t1 = tr.Times[i1];
                float u  = (t1 > t0) ? (tt - t0) / (t1 - t0) : 0f;

                var dq0 = tr.Keys[i0];
                var dq1 = tr.Keys[i1];
                var qr  = Slerp(dq0.qr, dq1.qr, u);
                var qd  = Vector4.Lerp(dq0.qd, dq1.qd, u);
                OrthoNormalize(ref qr, ref qd);
                local[b] = new DualQuat { qr = qr, qd = qd };
            }

            // 2) hierarchy: global = parent * local  (root-first order assumed)
            for (int b = 0; b < boneCount; b++)
            {
                int p = parentIndex[b];
                if (p < 0) { global[b] = local[b]; }
                else
                {
                    var pr = global[p].qr; var pd = global[p].qd;
                    var lr = local[b].qr;  var ld = local[b].qd;
                    var rr = QMul(pr, lr);
                    var rd = QMul(pr, ld) + QMul(pd, lr);
                    OrthoNormalize(ref rr, ref rd);
                    global[b] = new DualQuat { qr = rr, qd = rd };
                }
            }

            // 3) apply bind inverse: skin = global * bindInv
            for (int b = 0; b < boneCount; b++)
            {
                var g = global[b];
                var bi = bindInvDQ[b];
                var sr = QMul(g.qr, bi.qr);
                var sd = QMul(g.qr, bi.qd) + QMul(g.qd, bi.qr);
                OrthoNormalize(ref sr, ref sd);
                appendTo.Add(new DualQuat { qr = sr, qd = sd });
            }
        }
    }

    // ---------- Math / utils ----------

    private static DualQuat IdentityDQ() => new DualQuat { qr = new Vector4(0,0,0,1), qd = Vector4.Zero };

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

    private static Matrix4x4 ToM(global::Assimp.Matrix4x4 m) =>
        new Matrix4x4(m.A1,m.B1,m.C1,m.D1, m.A2,m.B2,m.C2,m.D2, m.A3,m.B3,m.C3,m.D3, m.A4,m.B4,m.C4,m.D4);

    private static DualQuat RigidToDQ(Quaternion r, Vector3 t)
    {
        var qr = Vector4.Normalize(new Vector4(r.X, r.Y, r.Z, r.W));
        var tq = new Vector4(t, 0f);
        var qd = 0.5f * QMul(tq, qr);
        return new DualQuat { qr = qr, qd = qd };
    }

    private static DualQuat MatRigidToDQ(Matrix4x4 m)
    {
        Matrix4x4.Decompose(m, out _, out var r, out var t);
        return RigidToDQ(r, t);
    }

    private static Vector3 ToV3(global::Assimp.Vector3D v) => new Vector3(v.X, v.Y, v.Z);
    private static Quaternion ToQ(global::Assimp.Quaternion q) => new Quaternion(q.X, q.Y, q.Z, q.W);

    private static void Walk(Node n, Action<Node> fn)
    {
        fn(n);
        foreach (var c in n.Children) Walk(c, fn);
    }
}
