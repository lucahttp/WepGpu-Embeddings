
const PCA = require('pca-js');

// Helper to simulate the new DimensionalityReducer.reduce
function reduce(embeddings, targetDim = 3) {
    if (embeddings.length < targetDim + 1) {
        return embeddings.map(e => e.slice(0, targetDim));
    }

    // 1. Get Eigenvectors
    const vectors = PCA.getEigenVectors(embeddings);
    
    // 2. Select Top K
    const topVectors = vectors.slice(0, targetDim).map(v => v.eigenvector);
    
    // 3. Project Data (Manual Matrix Multiplication)
    const N = embeddings.length;
    const K = targetDim;
    const projected = [];
    
    // Helper dot product
    const dot = (a, b) => a.reduce((sum, val, i) => sum + val * b[i], 0);

    for (let i = 0; i < N; i++) {
        const row = [];
        for (let j = 0; j < K; j++) {
            row.push(dot(embeddings[i], topVectors[j]));
        }
        projected.push(row);
    }

    return projected;
}

// Test with 8 items, 5 dimensions
const N = 8;
const D = 5;
const embeddings = [];
for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < D; j++) {
        row.push(Math.random());
    }
    embeddings.push(row);
}

try {
    console.log("Running manual reduction...");
    const result = reduce(embeddings, 3);
    console.log("Result shape:", result.length, "x", result[0].length);
    console.log("First point:", result[0]);
    if (result.length === N && result[0].length === 3) {
        console.log("SUCCESS: Output shape is correct.");
    } else {
        console.error("FAILURE: Incorrect output shape.");
    }
} catch (e) {
    console.error("FAILURE with error:", e);
}
