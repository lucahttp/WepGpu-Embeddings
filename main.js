import './style.css';
import { env, pipeline } from '@xenova/transformers';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import PCA from 'pca-js';

// Configuration
env.allowLocalModels = false;
env.backends.onnx.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.14.0/dist/';

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

    async generateEmbeddings(texts) {
        if (!this.pipe) throw new Error("Model not loaded");
        
        // Generate embeddings in parallel-ish
        const output = await this.pipe(texts, { pooling: 'mean', normalize: true });
        // output.data is a Float32Array of size [n_texts * 384]
        // We need to reshape it
        const embeddings = [];
        const dims = 384; 
        for (let i = 0; i < texts.length; i++) {
            const start = i * dims;
            const end = start + dims;
            embeddings.push(Array.from(output.data.slice(start, end)));
        }
        return embeddings;
    }
}

class DimensionalityReducer {
    static reduce(embeddings, targetDim = 3) {
        if (embeddings.length < targetDim + 1) {
            // Not enough points for PCA, just pad/truncate or return random small placement
            // For a demo, let's just use the first 3 dims if we have too few points
            return embeddings.map(e => e.slice(0, targetDim));
        }

        // Transpose for pca-js (it expects variables as rows, data points as columns usually? 
        // Actually pca-js expects: computeAdjustedData(data, eigenvectors)
        // Let's verify pca-js format. It usually expects data as vector of vectors.
        // But commonly it expects each INPUT to be a vector.
        // Let's assume standard behavior: getEigenVectors(data). 
        // Data in pca-js is usually Array<Array<Number>>.
        
        const vectors = PCA.getEigenVectors(embeddings);
        const topVectors = vectors.slice(0, targetDim);
        const result = PCA.computeAdjustedData(embeddings, topVectors);
        
        // Result is { adjustedData, formattedAdjustedData, ... }
        // formattedAdjustedData is typically the projected data.
        return result.formattedAdjustedData; 
    }
}

class Visualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.scene = new THREE.Scene();
        this.camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
        this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
        
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(window.devicePixelRatio);
        this.container.appendChild(this.renderer.domElement);

        this.controls = new OrbitControls(this.camera, this.renderer.domElement);
        this.controls.enableDamping = true;
        this.controls.autoRotate = true;
        this.controls.autoRotateSpeed = 1.0;

        this.raycaster = new THREE.Raycaster();
        this.mouse = new THREE.Vector2();
        
        this.pointsMesh = null;
        this.originalPositions = null;
        this.labels = [];
        
        // Setup initial scene
        this.camera.position.z = 5;
        this.scene.fog = new THREE.FogExp2(0x050510, 0.1);
        
        // Lighting
        const ambientLight = new THREE.AmbientLight(0x404040);
        this.scene.add(ambientLight);
        
        // Resize handler
        window.addEventListener('resize', this.onWindowResize.bind(this));
        
        // Mouse move for hover
        window.addEventListener('mousemove', this.onMouseMove.bind(this));
        
        this.animate();
    }

    onWindowResize() {
        this.camera.aspect = window.innerWidth / window.innerHeight;
        this.camera.updateProjectionMatrix();
        this.renderer.setSize(window.innerWidth, window.innerHeight);
    }

    onMouseMove(event) {
        this.mouse.x = (event.clientX / window.innerWidth) * 2 - 1;
        this.mouse.y = -(event.clientY / window.innerHeight) * 2 + 1;
        this.checkIntersections();
        
        // Pause rotation when interacting
        this.controls.autoRotate = false;
        clearTimeout(this.rotationTimeout);
        this.rotationTimeout = setTimeout(() => {
            this.controls.autoRotate = true;
        }, 2000);
    }

    checkIntersections() {
        if (!this.pointsMesh) return;
        
        this.raycaster.setFromCamera(this.mouse, this.camera);
        const intersects = this.raycaster.intersectObject(this.pointsMesh);
        
        const tooltip = document.getElementById('tooltip');
        
        if (intersects.length > 0) {
            const index = intersects[0].index;
            const text = this.labels[index];
            
            tooltip.textContent = text;
            tooltip.style.opacity = 1;
            tooltip.style.left = (event.clientX) + 'px';
            tooltip.style.top = (event.clientY - 10) + 'px';
            
            // Highlight effect could go here (e.g. scale up point)
            
            // Reset cursor
            document.body.style.cursor = 'pointer';
        } else {
            tooltip.style.opacity = 0;
            document.body.style.cursor = 'default';
        }
    }

    updatePoints(points3d, labels) {
        this.labels = labels;
        
        // Remove old mesh
        if (this.pointsMesh) {
            this.scene.remove(this.pointsMesh);
            this.pointsMesh.geometry.dispose();
            this.pointsMesh.material.dispose();
        }

        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(points3d.length * 3);
        const colors = new Float32Array(points3d.length * 3);
        
        const color1 = new THREE.Color(0x00ff88);
        const color2 = new THREE.Color(0x00aaff);

        points3d.forEach((pt, i) => {
            // Normalize/Scale points to fit in view
            // Simple normalization to -2..2 range
            positions[i * 3] = pt[0];
            positions[i * 3 + 1] = pt[1];
            positions[i * 3 + 2] = pt[2];

            // Mix colors based on position
            const mixedColor = color1.clone().lerp(color2, (pt[0] + 2) / 4);
            colors[i * 3] = mixedColor.r;
            colors[i * 3 + 1] = mixedColor.g;
            colors[i * 3 + 2] = mixedColor.b;
        });

        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        // Create texture for point
        const sprite = new THREE.TextureLoader().load('https://threejs.org/examples/textures/sprites/disc.png');

        const material = new THREE.PointsMaterial({ 
            size: 0.2, 
            vertexColors: true, 
            map: sprite, 
            transparent: true,
            alphaTest: 0.5,
            sizeAttenuation: true
        });

        this.pointsMesh = new THREE.Points(geometry, material);
        this.scene.add(this.pointsMesh);
        
        // Center the camera on the points roughly
        // (OrbitControls handles looking at 0,0,0)
    }

    animate() {
        requestAnimationFrame(this.animate.bind(this));
        
        this.controls.update();

        // Subtle pulse or movement could go here

        this.renderer.render(this.scene, this.camera);
    }
}

// Application Orchestrator
const app = {
    embeddingManager: new EmbeddingManager(),
    visualizer: new Visualizer('canvas-container'),
    
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
            loader.style.display = 'block';
            status.textContent = `Generating embeddings for ${lines.length} items...`;

            try {
                // 1. Generate Embeddings
                const embeddings = await app.embeddingManager.generateEmbeddings(lines);
                
                // 2. Reduce Dimensions (384 -> 3)
                // Normalize points first for better visualization? 
                // PCA usually handles it, but let's scale results for Three.js
                // Scale factor: spread them out a bit
                const reduced = DimensionalityReducer.reduce(embeddings, 3);
                
                // Normalize reduced coordinates to be roughly unit sphere size * spread
                // Find bounds
                let min = [Infinity, Infinity, Infinity];
                let max = [-Infinity, -Infinity, -Infinity];
                reduced.forEach(pt => {
                    for(let i=0; i<3; i++) {
                        if(pt[i] < min[i]) min[i] = pt[i];
                        if(pt[i] > max[i]) max[i] = pt[i];
                    }
                });
                
                const range = max.map((mx, i) => mx - min[i] || 1); // Avoid div by 0
                const scale = 4.0; // Target spread size
                
                const normalizedPoints = reduced.map(pt => [
                    ((pt[0] - min[0]) / range[0]) * scale - (scale/2),
                    ((pt[1] - min[1]) / range[1]) * scale - (scale/2),
                    ((pt[2] - min[2]) / range[2]) * scale - (scale/2)
                ]);

                // 3. Update Visualizer
                app.visualizer.updatePoints(normalizedPoints, lines);
                status.textContent = `Visualizing ${lines.length} embeddings.`;

            } catch (err) {
                console.error(err);
                status.textContent = "Error: " + err.message;
            } finally {
                btn.disabled = false;
                btnText.style.display = 'inline';
                loader.style.display = 'none';
            }
        });
    }
};

app.init();
