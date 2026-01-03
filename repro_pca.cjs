
const PCA = require('pca-js');

// Mock data: 8 items, 5 dimensions
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

// Helper
function transpose(matrix) {
    return matrix[0].map((col, i) => matrix.map(row => row[i]));
}

try {
    console.log("\n--- TEST: Raw Embeddings (8x5) ---");
    const vectors = PCA.getEigenVectors(embeddings);
    console.log(`Eigenvectors found: ${vectors.length}`);
    if (vectors.length > 0) {
        console.log("Vector 0 keys:", Object.keys(vectors[0]));
        // Check if it has 'vector' property
    }
    
    // Top 3
    const topVectors = vectors.slice(0, 3);
    
    // Try passing pure vectors if they are objects
    // In main.js it passes 'topVectors' which is array of objects.
    // Let's see if computeAdjustedData likes that.
    try {
        console.log("Calling computeAdjustedData(embeddings, topVectors)...");
        const result = PCA.computeAdjustedData(embeddings, topVectors);
        console.log("Success!");
    } catch (e) {
        console.log("Failed:", e.message);
    }

    // Try passing selected vectors (objects) individually? No, expected 2 args.
    
    // Try passing array of arrays (pure vectors)
    const pureVectors = topVectors.map(v => v.vector);
    try {
        console.log("Calling computeAdjustedData(embeddings, pureVectorsArray)...");
        const result = PCA.computeAdjustedData(embeddings, pureVectors);
        console.log("Success with pure vectors!");
    } catch (e) {
        console.log("Failed with pure vectors:", e.message);
    }
} catch (e) {
    console.error(e);
}

try {
    console.log("\n--- TEST: Transposed Embeddings (5x8) ---");
    const tEmbeddings = transpose(embeddings);
    // Columns are now "items" (8), Rows are variables (5).
    const vectors = PCA.getEigenVectors(tEmbeddings);
    console.log(`Eigenvectors found: ${vectors.length}`);
    
    const topVectors = vectors.slice(0, 3);
    
    try {
        console.log("Calling computeAdjustedData(tEmbeddings, topVectors)...");
        const result = PCA.computeAdjustedData(tEmbeddings, topVectors);
        console.log("Success!");
        console.log("Result formatted data:", result.formattedAdjustedData);
    } catch (e) {
        console.log("Failed:", e.message);
    }
} catch (e) {
    console.error(e);
}
