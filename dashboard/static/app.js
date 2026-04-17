document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const mockToggle = document.getElementById('mock-toggle');
    const hfSettings = document.getElementById('hf-settings');
    const btnReload = document.getElementById('btn-reload');
    const repoIdInput = document.getElementById('repo-id');
    const hfTokenInput = document.getElementById('hf-token');
    
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    const emptyState = document.getElementById('empty-state');
    const resultsState = document.getElementById('results-state');
    
    const oodSlider = document.getElementById('ood-slider');
    const oodVal = document.getElementById('ood-val');
    
    const tabs = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    // State
    let totalInferences = 0;
    let loadCounts = [0, 0, 0, 0, 0];
    const EXPERT_NAMES = [
        "E0: NIH ChestX-ray",
        "E1: ISIC 2019",
        "E2: Osteo",
        "E3: LUNA16",
        "E4: Pancreas 3D"
    ];

    // Toggle HF Settings
    mockToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            hfSettings.classList.add('hidden');
        } else {
            hfSettings.classList.remove('hidden');
        }
    });

    // Reload Engine
    btnReload.addEventListener('click', async () => {
        const btn = btnReload;
        btn.textContent = "Cargando...";
        btn.disabled = true;
        
        const formData = new FormData();
        formData.append("use_mock", mockToggle.checked);
        formData.append("repo_id", repoIdInput.value);
        if(hfTokenInput.value) formData.append("token", hfTokenInput.value);

        try {
            const res = await fetch('/api/reload_engine', { method: 'POST', body: formData });
            const data = await res.json();
            if(data.success) {
                alert("Motor recargado correctamente.");
            } else {
                alert("Error: " + data.error);
            }
        } catch (e) {
            alert("Error de conexión");
        } finally {
            btn.textContent = "Cargar Modelos Reales";
            btn.disabled = false;
        }
    });

    // OOD Slider
    oodSlider.addEventListener('input', (e) => {
        oodVal.textContent = e.target.value;
    });

    // Tabs
    tabs.forEach(tab => {
        tab.addEventListener('click', () => {
            tabs.forEach(t => t.classList.remove('active'));
            tabContents.forEach(c => c.classList.remove('active'));
            
            tab.classList.add('active');
            document.getElementById(`tab-${tab.dataset.tab}`).classList.add('active');
        });
    });

    // File Upload (Drag & Drop)
    uploadZone.addEventListener('click', () => fileInput.click());
    
    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });
    
    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });
    
    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    async function handleFile(file) {
        // UI Transition
        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        uploadZone.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch('/api/predict', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.success) {
                updateDashboard(data);
                
                // Update stats
                totalInferences++;
                document.getElementById('total-inferences').textContent = totalInferences;
                loadCounts[data.expert_id]++;
                updateLoadBalance();
                
            } else {
                alert("Error en la inferencia: " + data.error);
                emptyState.classList.remove('hidden');
            }
        } catch (error) {
            console.error("Error:", error);
            alert("Error de conexión al servidor.");
            emptyState.classList.remove('hidden');
        } finally {
            loadingIndicator.classList.add('hidden');
            uploadZone.classList.remove('hidden');
            fileInput.value = ""; // Reset input
        }
    }

    function updateDashboard(data) {
        // Show results
        resultsState.classList.remove('hidden');

        // Images
        document.getElementById('img-original').src = "data:image/png;base64," + data.display_image;
        if(data.heatmap_image) {
            document.getElementById('img-heatmap').src = "data:image/png;base64," + data.heatmap_image;
        }

        // Meta
        const dimStr = data.is_3d ? "3D" : "2D";
        document.getElementById('img-meta').textContent = 
            `${dimStr} | Original: ${data.original_shape.join('x')} | Adaptado: ${data.processed_shape.join('x')}`;

        // OOD
        const oodAlert = document.getElementById('ood-alert');
        // Check if user slider threshold overrides
        const userThreshold = parseInt(oodSlider.value) / 100;
        const isOOD = data.entropy > userThreshold;

        if (isOOD) {
            oodAlert.classList.remove('hidden');
            document.getElementById('ood-details').textContent = 
                `Entropía: ${data.entropy.toFixed(3)} nats • Umbral: ${userThreshold.toFixed(3)}`;
        } else {
            oodAlert.classList.add('hidden');
        }

        // Prediction
        const predLabel = document.getElementById('pred-label');
        const confBar = document.getElementById('conf-bar');
        const confVal = document.getElementById('conf-val');
        
        predLabel.textContent = data.class_label;
        const confPct = (data.confidence * 100).toFixed(1);
        confVal.textContent = `${confPct}%`;
        confBar.style.width = `${confPct}%`;
        
        if (confPct > 80) {
            predLabel.style.color = 'var(--color-green)';
            confBar.style.background = 'var(--color-green)';
        } else if (confPct > 50) {
            predLabel.style.color = 'var(--color-yellow)';
            confBar.style.background = 'var(--color-yellow)';
        } else {
            predLabel.style.color = 'var(--color-red)';
            confBar.style.background = 'var(--color-red)';
        }

        // Latency
        document.getElementById('total-ms').textContent = `${data.total_ms.toFixed(1)} ms`;
        document.getElementById('ms-pre').textContent = data.preprocess_ms.toFixed(1);
        document.getElementById('ms-router').textContent = data.router_ms.toFixed(1);
        document.getElementById('ms-expert').textContent = data.expert_ms.toFixed(1);

        // Expert Details
        document.getElementById('expert-name').textContent = `E${data.expert_id}: ${data.expert_name}`;
        document.getElementById('exp-arch').textContent = data.expert_arch;
        document.getElementById('exp-data').textContent = data.expert_dataset;
        document.getElementById('exp-classes').textContent = data.class_names.length;

        // Gating Scores
        const gatingList = document.getElementById('gating-list');
        gatingList.innerHTML = '';
        data.gating_scores.forEach((score, idx) => {
            const pct = (score * 100).toFixed(1);
            const isActive = idx === data.expert_id;
            const cls = isActive ? 'gating-row active' : 'gating-row';
            const color = isActive ? 'var(--accent-blue)' : '#1e293b';
            
            gatingList.innerHTML += `
                <div class="${cls}">
                    <div class="g-label"><div class="dot"></div>${EXPERT_NAMES[idx]}</div>
                    <div class="bar-track">
                        <div class="bar-fill" style="width: ${pct}%; background: ${color}"></div>
                    </div>
                    <div class="g-val">${pct}%</div>
                </div>
            `;
        });

        // Probabilities Table
        const probsTable = document.querySelector('#probs-table tbody');
        probsTable.innerHTML = '';
        const probsArr = data.class_names.map((name, i) => ({
            name: name,
            prob: data.all_class_probs[i]
        })).sort((a, b) => b.prob - a.prob);

        probsArr.forEach(item => {
            const pct = (item.prob * 100).toFixed(1);
            probsTable.innerHTML += `
                <tr>
                    <td>${item.name}</td>
                    <td style="font-family: monospace;">${item.prob.toFixed(3)}</td>
                    <td>
                        <div class="prob-bar-container">
                            <div class="prob-bar-fill" style="width: ${pct}%"></div>
                        </div>
                    </td>
                </tr>
            `;
        });

        // History Table Update
        const historyTbody = document.getElementById('history-tbody');
        const oodBadge = isOOD ? '<span class="badge-table badge-ood">Sí</span>' : '<span class="badge-table badge-ok">No</span>';
        
        // prepend to show latest first
        const newRow = document.createElement('tr');
        newRow.innerHTML = `
            <td>${totalInferences}</td>
            <td>${data.is_3d ? 'Volumen 3D' : 'Imagen 2D'}</td>
            <td style="font-weight: 600;">${data.class_label}</td>
            <td style="font-family: monospace;">${(data.confidence * 100).toFixed(1)}%</td>
            <td>E${data.expert_id}: ${data.expert_name.split(' ')[0]}</td>
            <td>${oodBadge}</td>
        `;
        historyTbody.prepend(newRow);
    }

    function updateLoadBalance() {
        const total = loadCounts.reduce((a, b) => a + b, 0);
        if (total === 0) return;

        let maxF = 0;
        let minF = Infinity;
        const fractions = loadCounts.map(c => {
            const f = c / total;
            if (f > 0) {
                if (f > maxF) maxF = f;
                if (f < minF) minF = f;
            }
            return f;
        });

        const ratio = minF === Infinity || minF === 0 ? 1.0 : maxF / minF;
        
        const ratioEl = document.getElementById('ratio-val');
        ratioEl.textContent = ratio.toFixed(2);
        if (ratio > 1.30) {
            ratioEl.style.color = 'var(--color-red)';
        } else {
            ratioEl.style.color = 'var(--color-green)';
        }

        const barsContainer = document.getElementById('balance-bars');
        barsContainer.innerHTML = '';
        
        fractions.forEach((f, idx) => {
            const isMax = f === maxF;
            const color = !isMax ? 'var(--accent-blue)' : (ratio > 1.30 ? 'var(--color-red)' : 'var(--color-green)');
            const pct = (f * 100).toFixed(1);
            
            barsContainer.innerHTML += `
                <div class="b-row">
                    <div class="b-label">${EXPERT_NAMES[idx].split(':')[1].trim()}</div>
                    <div class="b-track">
                        <div class="b-fill" style="width: ${pct}%; background: ${color}"></div>
                    </div>
                    <div class="b-val">${f.toFixed(3)}</div>
                </div>
            `;
        });
    }
});
