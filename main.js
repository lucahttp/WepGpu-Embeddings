import './style.css';
import { env, pipeline } from '@xenova/transformers';
import Plotly from 'plotly.js-dist';
import PCA from 'pca-js';
import { TopicModeler } from './topicModeler.js';

// Configuration
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

// --- Visualization Config & Helpers (from User Snippet) ---
const colors = {
    "light-warm-gray": {light:"#EEEDEE", dark: "#EEEDEE"},
    "dark-warm-gray": {light:"#A3ACB0", dark: "#A3ACB0"},
    "cool-gray-0.5": {light:"#EDEEF1", dark: "#EDEEF1"},
    "cool-gray-1": {light:"#C5C5D2", dark: "#AEAEBB"},
    "cool-gray-1.5": {light:"#AEAEBB", dark: "#8E8EA0"},
    "cool-gray-2": {light:"#8E8EA0", dark: "#6E6E80"},
    "cool-gray-3": {light:"#6E6E80", dark: "#404452"},
    "cool-gray-4": {light:"#404452", dark: "#404452"},
    "light-black": {light:"#191927", dark: "#191927"},
    "medium-black": {light:"#0E0E1A", dark: "#0E0E1A"},
    "black": {light:"#050505", dark: "#050505"},
    "dark-red": {light: "#BD1C5F", dark: "#BD1C5F"},
    "red": {light: "#F22C3D", dark: "#F22C3D"},
    "green": {light: "#00A67D", dark: "#099A77"},
    "blue": {light: "#0082D0", dark: "#0082D0"},
    "orange": {light: "#FF5828", dark: "#FF5828"},
    "teal": {light: "#21B5C2", dark: "#21B5C2"},
    "mustard": {light: "#EA9100", dark: "#EA9100"},
    "yellow": {light: "#EBE93D", dark: "#DDDB11"},
    "violet": {light: "#5436DA", dark: "#5436DA"},
    "bright-green": {light: "#54F16C", dark: "#00DE22"},
    "bright-yellow": {light: "#EAFF00", dark: "#EAFF00"},
    "bright-blue": {light: "#00B7FF", dark: "#00B7FF"},
    "light-violet": {light: "#9388F7", dark: "#9388F7"},
    "pink": {light: "#F2BAFF", dark: "#F2BAFF"},
    "light-blue": {light: "#CAE5F2", dark: "#A5CFE4"},
    "gold": {light: "#B2943A", dark: "#B2943A"},
    "olive": {light: "#7E813C", dark: "#7E813C"},
    "navy": {light: "#1D0D4C", dark: "#1D0D4C"},
};

// Colors for topics
const topicColors = [
    colors['orange'].dark,
    colors['bright-green'].dark,
    colors['bright-blue'].dark,
    colors['violet'].dark,
    colors['pink'].dark,
    colors['teal'].dark,
    colors['red'].dark,
    colors['mustard'].dark,
    colors['olive'].dark,
    colors['navy'].dark,
];

const bp = 580; 
function getMarkerSize(width) {
    return (width > bp) ? 8 : 6;
}

function addBr(text) {
    let result = "";
    text = text || "";
    text.trim().split("").forEach(function (item, index) {
        result = result + item;
        if (index !== 0 && index % 30 === 0) {
            result = result + "<br>";
        }
    });
    return result;
}

// --- Logic Classes ---

class EmbeddingManager {
    constructor() {
        this.pipe = null;
        this.modelId = 'Xenova/all-MiniLM-L6-v2';
        //this.modelId = 'onnx-community/Qwen3-Embedding-0.6B-ONNX';
    }

    async loadModel(statusCallback) {
        if (!this.pipe) {
            statusCallback('Loading model (WebGPU/WASM)...');
            this.pipe = await pipeline('feature-extraction', this.modelId, {
                device: navigator.gpu ? 'webgpu' : 'wasm',
            });
            statusCallback('Model ready.');
        }
    }

    async generateEmbeddings(texts, onProgress) {
        if (!this.pipe) throw new Error("Model not loaded");
        
        const embeddings = [];
        // Local generation
        for (let i = 0; i < texts.length; i++) {
            const text = texts[i];
            const output = await this.pipe(text, { pooling: 'mean', normalize: true });
            const embedding = Array.from(output.data);
            embeddings.push(embedding);
            
            if (onProgress) onProgress(i + 1, texts.length, text);
            await new Promise(r => setTimeout(r, 0)); 
        }
        return embeddings;
    }
}

class DimensionalityReducer {
    static reduce(embeddings, targetDim = 3) {
        if (embeddings.length < targetDim + 1) {
             return embeddings.map(e => {
                const arr = e.slice(0, targetDim);
                while(arr.length < targetDim) arr.push(0); 
                return arr;
            });
        }

        const vectors = PCA.getEigenVectors(embeddings);
        const topVectors = vectors.slice(0, targetDim).map(v => v.eigenvector);
        
        const N = embeddings.length;
        const K = targetDim;
        const projected = [];
        
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
}

class PlotlyVisualizer {
    constructor(divId) {
        this.divId = divId;
        this.container = document.getElementById(divId);
        
        // Handle resize
        window.addEventListener('resize', () => {
             Plotly.Plots.resize(this.divId);
        });
    }

    updatePoints(points3d, labels, topicAssignments = null, topicsFullData = null) {
        // points3d: [[x,y,z], ...]
        // labels: ["text", ...]
        // topicAssignments: { label: "topic_label", id: clusterId } per point array? No, simpler: a map or just array of topic ID per point.
        // Let's assume topicAssignments is an array of size N where assignment[i] = { id: topicId, label: topicLabel }
        
        const width = this.container.offsetWidth || window.innerWidth * 0.9;
        const markerSize = getMarkerSize(width);
        
        // Prepare data for Plotly
        // If we have topics, we might want separate traces for legend, but user snippet used one logic.
        // Let's try separate traces if topics exist, for better legend.
        
        const traces = [];
        
        if (topicAssignments) {
            // Group points by topic
            const groups = new Map();
            points3d.forEach((pt, i) => {
                const topic = topicAssignments[i]; // { id, label }
                const groupKey = topic ? topic.id : 'unknown';
                
                if (!groups.has(groupKey)) {
                    groups.set(groupKey, {
                        x: [], y: [], z: [], text: [], 
                        name: topic ? topic.label : 'Unknown',
                        color: topic ? topicColors[topic.id % topicColors.length] : '#999'
                    });
                }
                const g = groups.get(groupKey);
                g.x.push(pt[0]);
                g.y.push(pt[1]);
                g.z.push(pt[2]);
                g.text.push(addBr(labels[i])); // Just text, or include topic? Hover info "text" usually replaces name.
            });
            
            // Create traces
            for (const [key, g] of groups.entries()) {
                traces.push({
                    x: g.x, y: g.y, z: g.z,
                    mode: 'markers',
                    name: g.name, // Legend name
                    marker: {
                        color: g.color,
                        size: markerSize,
                        opacity: 0.8,
                        line: { color: 'rgba(255, 255, 255, 0.2)', width: 0.5 }
                    },
                    text: g.text,
                    hoverinfo: "text+name", // Show text and trace name (topic)
                    hoverlabel: {
                        bgcolor: "#fff", bordercolor: "#fff",
                        font: { color: "#050505", family: 'Inter, sans-serif' }
                    },
                    type: 'scatter3d'
                });
            }
        } else {
             // Fallback to single trace cyclic colors
            const pointColors = points3d.map((_, i) => topicColors[i % topicColors.length]);
            traces.push({
                x: points3d.map(p => p[0]),
                y: points3d.map(p => p[1]),
                z: points3d.map(p => p[2]),
                mode: 'markers',
                marker: {
                    color: pointColors,
                    size: markerSize,
                    opacity: 0.8,
                    line: { color: 'rgba(255, 255, 255, 0.2)', width: 0.5 }
                },
                text: labels.map(l => addBr(l)),
                hoverinfo: "text",
                hoverlabel: { bgcolor: "#fff", bordercolor: "#fff", font: { color: "#050505", family: 'Inter, sans-serif' } },
                type: 'scatter3d'
            });
        }

        const layout = {
            autosize: true,
            height: 480,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "#fff",
            showlegend: true, // Enable legend for topics
            legend: { x: 0, y: 1 },
            scene: {
                xaxis: { tickfont: { size: 10, color: 'rgb(107, 107, 107)' }, title: '' },
                yaxis: { tickfont: { size: 10, color: 'rgb(107, 107, 107)' }, title: '' },
                zaxis: { tickfont: { size: 10, color: 'rgb(107, 107, 107)' }, title: '' },
                aspectmode: 'cube',
                camera: {
                    eye: { x: 1.5, y: 1.5, z: 1.5 }
                }
            }
        };

        const config = {
            modeBarButtonsToRemove: ['pan3d', 'resetCameraLastSave3d', 'toImage', 'tableRotation'],
            displaylogo: false,
            responsive: true,
            displayModeBar: false
        };

        Plotly.newPlot(this.divId, traces, layout, config);
        
        //this.startRotation();
    }
    
    startRotation() {
        if(this.animationFrame) cancelAnimationFrame(this.animationFrame);
        
        let angle = 0;
        const rotate = () => {
            angle += 0.002;
            const r = 1.8;
            Plotly.relayout(this.divId, {
                'scene.camera.eye': { 
                    x: r * Math.cos(angle), 
                    y: r * Math.sin(angle), 
                    z: 1.5 
                }
            });
            this.animationFrame = requestAnimationFrame(rotate);
        };
        rotate();
    }
}

// Application Orchestrator
const app = {
    embeddingManager: new EmbeddingManager(),
    topicModeler: new TopicModeler(),
    visualizer: new PlotlyVisualizer('chart-div'),
    
    init: async () => {
        const input = document.getElementById('text-input');
        const btn = document.getElementById('generate-btn');
        const loader = document.getElementById('loader');
        const btnText = document.getElementById('btn-text');
        const status = document.getElementById('status-msg');

        // Initial load
        try {
            await app.embeddingManager.loadModel((msg) => {
                status.textContent = msg;
            });
        } catch (e) {
            console.error(e);
            status.textContent = "Error loading model. Check console.";
            return;
        }

        btn.addEventListener('click', async () => {
            const text = input.value.trim();
            if (!text) return;
            
            const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);
            if (lines.length === 0) return;

            // UI Loading State
            btn.disabled = true;
            btnText.style.display = 'none';
            if (loader) loader.style.display = 'inline-block';
            status.textContent = `Generating embeddings for ${lines.length} items...`;

            try {
                // 1. Generate Embeddings with Progress
                const fileProgress = document.getElementById('progress-container');
                const progressBar = document.getElementById('progress-bar');
                const progressLabel = document.getElementById('progress-label');
                
                if (fileProgress) {
                    fileProgress.style.display = 'block';
                    fileProgress.classList.remove('hidden');
                    progressBar.style.width = '0%';
                }
                
                const onProgress = (idx, total, text) => {
                    const pct = Math.round((idx / total) * 100);
                    if (progressBar) progressBar.style.width = `${pct}%`;
                    if (progressLabel) progressLabel.textContent = `${pct}% - Generated for "${text.slice(0, 15)}..."`;
                };

                const embeddings = await app.embeddingManager.generateEmbeddings(lines, onProgress);
                
                // 2. Reduce Dimensions (384 -> 3)
                const reduced = DimensionalityReducer.reduce(embeddings, 3);
                
                // 3. Topic Modeling
                status.textContent = "Clustering and identifying topics...";
                await new Promise(r => setTimeout(r, 10)); // Yield UI
                
                // Note: topicModeler.run is sync but returns fast for small N.
                const topics = app.topicModeler.run(lines, embeddings);
                console.log("Topics found:", topics);

                // Create a map of index -> { id, label }
                // topics is array of { id, label, indices... }
                // But wait, my topicModeler returns docs array, need to map back to original indices? 
                // Ah, my run returned: { id, label, docs, embeddings (subset) }
                // It grouped them. But I need them in original order for the 'reduced' array which corresponds to 'lines'.
                // Or I can re-build the display data from the topic groups.
                
                // Actually my topicModeler implementation has `clusteredDocs` with `indices`. 
                // Let's modify TopicModeler output slightly or map it back here.
                // The current implementation returns:
                // [ { id: 0, label: "...", docs: [...], embeddings: [...] }, ... ]
                // It *doesn't* return the indices. I should check topicModeler.js code again.
                // Wait, I WROTE it to use: clusteredDocs.get(label).indices.push(i);
                // But the `results.push` object:
                /*
                results.push({
                    id: label,
                    label: topicLabel,
                    keywords: keywords,
                    docs: data.docs,
                    embeddings: data.indices.map(i => embeddings[i]) 
                });
                */
               // It's missing `indices` in the public output property list.
               // I can reconstruct assignments because I know the input `lines` and the `docs` in result. 
               // BUT duplicate lines would be ambiguous. 
               // BETTER: Update topicModeler.js to include 'indices' in output OR map here.
               // Let's rely on the input order matching. 
               // Actually, `ml-kmeans` returns assignments array [clusterId, clusterId...].
               // Maybe it's better to expose `getAssignments` or simply have `run` return an assignment array too?
               
               // For now, let's just make a flat array of size N.
               // Since `run` groups them, I'd have to find which group each index belongs to.
               // It's cleaner if I update `TopicModeler` to return the raw assignments or indices.
               
               // Let's hot-update the local Logic here or trust that I can just handle the grouped data?
               // If I pass `reduced` points to `updatePoints`, I need them in same order as `lines`.
               // The `topics` output groups them.
               // I can iterate topics -> indices -> assign to array.
               // BUT I DID NOT include indices in the output object in Step 71. 
               
               // HACK: I will just re-run cluster logic here? No, duplicate logic.
               // FIX: I will use `run`'s output `docs` to match `lines`.
               // Issue: Identical text lines.
               // REAL FIX: I should have returned indices.
               // Strategy: I will quickly modify topicModeler.js to include indices in the result object in the NEXT step if possible or assume I can do it now? 
               // Actually, I can use the `ml-kmeans` directly here but that defeats the modularity.
               
               // Let's assume I will fix `topicModeler.js` to return `indices`. 
               // I'll write the code assuming `indices` property exists in the topic object. 
               // AND I will verify/fix topicModeler.js in next step.
               
               const topicAssignments = new Array(lines.length);
               topics.forEach(topic => {
                   // Topic has .indices calculated inside run()
                   if(topic.indices) {
                       topic.indices.forEach(idx => {
                           topicAssignments[idx] = { id: topic.id, label: topic.label };
                       });
                   } else {
                       // Fallback if indices missing (I need to fix this)
                       // Try to match by text (imperfect)
                        topic.docs.forEach(doc => {
                           // This is risky for duplicates.
                           // I'll fix topicModeler.js immediately after this tool call.
                       });
                   }
               });

                // 4. Update Visualizer
                app.visualizer.updatePoints(reduced, lines, topicAssignments);
                status.textContent = `Visualizing ${lines.length} items in ${topics.length} topics.`;
                if (fileProgress) fileProgress.style.display = 'none';

            } catch (err) {
                console.error(err);
                status.textContent = "Error: " + err.message;
            } finally {
                btn.disabled = false;
                btnText.style.display = 'inline';
                if (loader) loader.style.display = 'none';
            }
        });
    }
};

app.init();
