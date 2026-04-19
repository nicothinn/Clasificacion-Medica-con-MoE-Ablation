document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const uploadZone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');
    const loadingIndicator = document.getElementById('loading-indicator');
    const emptyState = document.getElementById('empty-state');
    const resultsState = document.getElementById('results-state');
    const systemLog = document.getElementById('system-log');

    function agregarAlLog(mensaje, type = 'info') {
        if (!systemLog) return;
        const now = new Date();
        const timeStr = now.toLocaleTimeString([], { hour12: false });
        
        const div = document.createElement('div');
        div.className = `log-entry log-${type}`;
        
        const timeSpan = document.createElement('span');
        timeSpan.className = 'log-time';
        timeSpan.textContent = `[${timeStr}]`;
        
        const msgSpan = document.createElement('span');
        msgSpan.className = 'log-msg';
        msgSpan.textContent = ` ${mensaje}`;
        
        div.appendChild(timeSpan);
        div.appendChild(msgSpan);
        
        systemLog.appendChild(div);
        systemLog.scrollTop = systemLog.scrollHeight;
    }
    
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

    // OOD Threshold — fixed at 85% of max entropy (calibrated value)
    const UMBRAL_OOD_DEFAULT = 0.85;

    function calcularEntropia(gatingScores) {
        let entropia = 0;
        for (let i = 0; i < gatingScores.length; i++) {
            const p = gatingScores[i];
            if (p > 0) {
                entropia -= p * Math.log(p);
            }
        }
        return entropia;
    }

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
            handleFiles(e.dataTransfer.files);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFiles(e.target.files);
        }
    });

    async function handleFiles(fileList) {
        const files = Array.from(fileList);
        
        // Validacion visual de MHD + RAW
        const hasMhd = files.some(f => f.name.toLowerCase().endsWith('.mhd'));
        const hasRaw = files.some(f => f.name.toLowerCase().endsWith('.raw') || f.name.toLowerCase().endsWith('.zraw'));
        if (hasMhd && !hasRaw) {
            alert("Has seleccionado un archivo .mhd pero falta el .raw/.zraw correspondiente. Por favor, asegúrate de seleccionar ambos al mismo tiempo.");
            fileInput.value = "";
            return;
        }

        // UI Transition
        emptyState.classList.add('hidden');
        resultsState.classList.add('hidden');
        uploadZone.classList.add('hidden');
        loadingIndicator.classList.remove('hidden');

        systemLog.innerHTML = '';
        const mainFile = files.find(f => !f.name.toLowerCase().endsWith('.raw') && !f.name.toLowerCase().endsWith('.zraw')) || files[0];
        const shortName = mainFile.name.length > 25 ? mainFile.name.substring(0, 22) + '...' : mainFile.name;
        agregarAlLog(`Procesando ${files.length} archivo(s): ${shortName}`, 'info');
        
        setTimeout(() => {
            const is3D = files.some(f => f.name.toLowerCase().match(/\.(nii|nii\.gz|mha|mhd)$/));
            agregarAlLog(`Extrayendo features ${is3D ? '3D' : '2D'}...`, 'info');
        }, 300);

        const formData = new FormData();
        files.forEach(f => {
            formData.append("files", f); // Cambiado a 'files' plural
        });
        
        // Debugging frontend: verificar qué archivos van en el FormData
        const archivosAdjuntos = formData.getAll("files").map(f => f.name);
        console.log("Archivos adjuntos en FormData:", archivosAdjuntos);
        
        // Get clinical source if specified (or default to 'unknown')
        const sourceSelect = document.getElementById('data-source');
        if (sourceSelect) {
            formData.append("source", sourceSelect.value);
        } else {
            formData.append("source", "unknown");
        }

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

    const ROUTING_PATHS = [
        "M 100 35 L 30 95",
        "M 100 35 L 65 95",
        "M 100 35 L 100 95",
        "M 100 35 L 135 95",
        "M 100 35 L 170 95"
    ];

    function updateDashboard(data) {
        // Show results
        resultsState.classList.remove('hidden');

        // Animate Routing Diagram
        const activePath = document.getElementById('active-routing-path');
        if (activePath) {
            activePath.classList.remove('animate');
            setTimeout(() => {
                activePath.setAttribute('d', ROUTING_PATHS[data.expert_id]);
                activePath.classList.add('animate');
                
                // Highlight active node
                for (let i = 0; i < 5; i++) {
                    const nodeCircle = document.getElementById(`node-e${i}`);
                    if (nodeCircle) {
                        const nodeText = nodeCircle.nextElementSibling;
                        if (i === data.expert_id) {
                            nodeText.style.fill = '#8B0000';
                            nodeText.style.fontWeight = '700';
                            nodeCircle.style.stroke = '#8B0000';
                            nodeCircle.style.strokeWidth = '2.5';
                        } else {
                            nodeText.style.fill = '#1F3330';
                            nodeText.style.fontWeight = '600';
                            nodeCircle.style.stroke = '';
                            nodeCircle.style.strokeWidth = '';
                        }
                    }
                }
            }, 50);
        }

        // Images
        document.getElementById('img-original').src = "data:image/png;base64," + data.display_image;
        if(data.heatmap_image) {
            document.getElementById('img-heatmap').src = "data:image/png;base64," + data.heatmap_image;
        }

        // Meta
        const dimStr = data.is_3d ? "3D" : "2D";
        document.getElementById('img-meta').textContent = 
            `${dimStr} | Original: ${data.original_shape.join('x')} | Adaptado: ${data.processed_shape.join('x')}`;

        // OOD — calculate entropy using JS
        const oodAlert = document.getElementById('ood-alert');
        const calculatedEntropy = calcularEntropia(data.gating_scores);
        const isOOD = calculatedEntropy > UMBRAL_OOD_DEFAULT;
        
        const predCard = document.getElementById('pred-card');

        if (isOOD) {
            oodAlert.classList.remove('hidden');
            document.getElementById('ood-details').textContent = 
                `Entropía: ${calculatedEntropy.toFixed(3)} nats • Umbral: ${UMBRAL_OOD_DEFAULT.toFixed(3)}`;
            agregarAlLog(`Entropía: ${calculatedEntropy.toFixed(3)} nats (ALERTA OOD)`, 'error');
            
            // Block prediction card
            predCard.style.opacity = '1';
            document.getElementById('pred-label').textContent = 'IMAGEN NO COMPATIBLE';
            document.getElementById('pred-label').style.color = 'var(--color-red)';
            document.getElementById('pred-label').style.fontSize = '1.2rem';
            
            // Show explainer, hide confidence
            document.getElementById('ood-explainer').classList.remove('hidden');
            document.getElementById('conf-wrapper').classList.add('hidden');
        } else {
            oodAlert.classList.add('hidden');
            agregarAlLog(`Entropía: ${calculatedEntropy.toFixed(3)} nats`, 'success');
            
            // Reset prediction card
            predCard.style.opacity = '1';
            document.getElementById('pred-label').style.fontSize = '';
            document.getElementById('ood-explainer').classList.add('hidden');
            document.getElementById('conf-wrapper').classList.remove('hidden');
        }

        setTimeout(() => {
            agregarAlLog(`Ruteo completado -> Experto ${data.expert_id}`, 'info');
            agregarAlLog(`Inferencia en ${data.expert_arch}...`, 'info');
        }, 500);

        setTimeout(() => {
            agregarAlLog(`Predicción: ${data.class_label} (${(data.confidence * 100).toFixed(1)}%)`, 'success');
            agregarAlLog(`Latencia total: ${data.total_ms.toFixed(1)} ms`, 'warning');
        }, 800);

        // Prediction
        const predLabel = document.getElementById('pred-label');
        const confBar = document.getElementById('conf-bar');
        const confVal = document.getElementById('conf-val');
        
        if (!isOOD) {
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
