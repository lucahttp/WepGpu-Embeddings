import './style.css';
import { env, pipeline } from '@xenova/transformers';
import Plotly from 'plotly.js-dist';
import PCA from 'pca-js';

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

// Default categories to cycle through if no category logic exists
const defaultColors = [
    colors['orange'].dark,
    colors['bright-green'].dark,
    colors['bright-blue'].dark,
    colors['violet'].dark,
    colors['pink'].dark,
    colors['teal'].dark,
    colors['red'].dark
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
            // If not enough points for PCA, just slice (fallback)
            // Or better, error or pad. Slicing is okay for simple fallback.
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

    updatePoints(points3d, labels) {
        // points3d: [[x,y,z], ...]
        // labels: ["text", ...]
        
        const width = this.container.offsetWidth || window.innerWidth * 0.9;
        const markerSize = getMarkerSize(width);

        // We will create one trace for simplicity, but coloring each point differently 
        // OR mimicking the category style if we can guess. 
        // Since we don't have categories, let's treat each point as a separate "trace" 
        // creates too many legend items.
        // Instead, let's make one trace with an array of colors.
        
        // Assign colors cyclically
        const pointColors = points3d.map((_, i) => defaultColors[i % defaultColors.length]);

        const trace = {
            x: points3d.map(p => p[0]),
            y: points3d.map(p => p[1]),
            z: points3d.map(p => p[2]),
            mode: 'markers',
            marker: {
                color: pointColors,
                size: markerSize,
                opacity: 0.8,
                line: {
                    color: 'rgba(255, 255, 255, 0.2)',
                    width: 0.5,
                }
            },
            text: labels.map(l => addBr(l)),
            hoverinfo: "text",
            hoverlabel: {
                bgcolor: "#fff",
                bordercolor: "#fff",
                font: {
                    color: "#050505",
                    family: 'Inter, sans-serif'
                },
            },
            type: 'scatter3d'
        };

        const layout = {
            autosize: true,
            height: 480,
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "#fff",
            showlegend: false, 
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

        Plotly.newPlot(this.divId, [trace], layout, config);
        
        // Start rotation animation
        //this.startRotation();
    }
    
    startRotation() {
        if(this.animationFrame) cancelAnimationFrame(this.animationFrame);
        
        let angle = 0;
        const rotate = () => {
            angle += 0.002;
            const r = 1.8;
            // Use Plotly.animate or relayout for camera
            // relayout is often smoother for just camera
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
                
                // 3. Update Visualizer
                // No normalization needed as PCA output is usually centered around 0 and Plotly auto-scales axis
                // But user snippet had some logic about traces. We just pass raw PCA output?
                // PCA output scales with variance. Plotly handles autosizing well.
                
                app.visualizer.updatePoints(reduced, lines);
                status.textContent = `Visualizing ${lines.length} embeddings.`;
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
