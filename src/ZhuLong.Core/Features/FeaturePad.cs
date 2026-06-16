namespace ZhuLong.Core.Features;

/// <summary>特征维度常量 — 与 config model.feature_dim_5min 一致。</summary>
public static class FeatureConstants
{
    public const int BaseFeatureDim = 22;
    public const int ModelFeatureDim = 30;
}

public static class FeaturePad
{
    /// <summary>将 (seqLen, 22) + 可选 MTF(6) 填充为 (seqLen, 30)。</summary>
    public static float[,] ToModelDim(
        float[,] seq,
        float[,]? mtf = null,
        int targetDim = FeatureConstants.ModelFeatureDim)
    {
        var rows = seq.GetLength(0);
        var cols = seq.GetLength(1);
        if (cols >= targetDim) return seq;

        var padded = new float[rows, targetDim];
        for (var i = 0; i < rows; i++)
        {
            for (var j = 0; j < cols; j++)
                padded[i, j] = seq[i, j];

            if (mtf is not null && mtf.GetLength(0) == rows)
            {
                var mtfCols = Math.Min(mtf.GetLength(1), targetDim - cols);
                for (var j = 0; j < mtfCols; j++)
                    padded[i, cols + j] = mtf[i, j];
            }
        }
        return padded;
    }
}
