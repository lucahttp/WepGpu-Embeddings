
const PCA = require('pca-js');

// Mock data: 8 items, 5 dimensions (simulating 384)
const N = 8;
const D = 5;

// Create random data
const embeddings = [];
for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < D; j++) {
        row.push(Math.random());
    }
    embeddings.push(row);
}

console.log(`Input shape: ${embeddings.length} x ${embeddings[0].length}`);

try {
    console.log("Attempting PCA.getEigenVectors with raw embeddings...");
    const vectors = PCA.getEigenVectors(embeddings);
    console.log(`Vectors found: ${vectors.length}`);
    
    const targetDim = 3;
    const topVectors = vectors.slice(0, targetDim);
    console.log(`Top vectors sliced: ${topVectors.length}`);
    
    console.log("Attempting PCA.computeAdjustedData...");
    const result = PCA.computeAdjustedData(embeddings, topVectors);
    console.log("Success!");
    console.log(result.formattedAdjustedData);
} catch (e) {
    console.error("Error with raw embeddings:", e.message);
}

// Try transpose?
function transpose(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

try {
    console.log("\nAttempting with Transposed data...");
    const tEmbeddings = transpose(embeddings);
    console.log(`Transposed shape: ${tEmbeddings.length} x ${tEmbeddings[0].length}`);

    const vectors = PCA.getEigenVectors(tEmbeddings);
    console.log(`Vectors found: ${vectors.length}`);
    
    const targetDim = 3;
    const topVectors = vectors.slice(0, targetDim);
    
    const result = PCA.computeAdjustedData(tEmbeddings, topVectors);
    console.log("Success with transpose!");
    // Result will be D x N? Or N x D?
    // original samples were columns in tEmbeddings.
    // result.formattedAdjustedData is the projected data.
    // If transposed, variables = rows = 384 (here 5).
    // Samples = columns = 8.
    // We want to reduce variables (dimensions) to 3? 
    // Wait, PCA reduces dimensionality of the *data set*.
    // Usually we want to reduce features.
} catch (e) {
    console.error("Error with transpose:", e.message);
}
