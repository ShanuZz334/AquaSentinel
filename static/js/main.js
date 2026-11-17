document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentPreset = 'ghost_net_reef.jpg';
    let uploadedFile = null;
    let analysisResults = {
        raw: '/api/samples/ghost_net_reef.jpg',
        enhanced: null,
        grounded: null,
        heatmap: null
    };

    // DOM Elements
    const tabButtons = document.querySelectorAll('.nav-item');
    const tabPanes = document.querySelectorAll('.tab-pane');
    const presetsContainer = document.getElementById('presets-container');
    const fileInput = document.getElementById('image-upload');
    const fileStatus = document.getElementById('file-status');
    const promptInput = document.getElementById('mission-prompt');
    const runBtn = document.getElementById('run-mission-btn');
    const loader = document.getElementById('loader');
    const displayImg = document.getElementById('display-image');
    const viewButtons = document.querySelectorAll('.control-btn');
    const directiveChips = document.querySelectorAll('.query-pill');
    const enhancementToggle = document.getElementById('enhancement-toggle');

    // Report Elements
    const threatStatusEl = document.getElementById('report-threat-status');
    const hazardScoreEl = document.getElementById('hazard-score');
    const narrativeEl = document.getElementById('report-narrative');
    const entitiesTbody = document.getElementById('entities-tbody');
    const protocolTextEl = document.getElementById('protocol-text');
    const castPctEl = document.getElementById('cast-pct');

    // 1. Tab Navigation
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.remove('active'));

            btn.classList.add('active');
            const targetPane = document.getElementById(btn.dataset.tab);
            if (targetPane) targetPane.classList.add('active');

            if (btn.dataset.tab === 'benchmarks') {
                loadBenchmarkCharts();
            }
        });
    });

    // 2. Preset Target Selection
    presetsContainer.querySelectorAll('.target-card').forEach(card => {
        card.addEventListener('click', () => {
            presetsContainer.querySelectorAll('.target-card').forEach(c => c.classList.remove('active'));
            card.classList.add('active');

            currentPreset = card.dataset.filename;
            uploadedFile = null;
            fileStatus.textContent = 'Standard target active';
            fileInput.value = '';

            displayImg.src = `/api/samples/${currentPreset}`;
            const labImg = document.getElementById('lab-preview-img');
            if (labImg) labImg.src = `/api/samples/${currentPreset}`;

            analysisResults.raw = `/api/samples/${currentPreset}`;
            analysisResults.enhanced = null;
            analysisResults.grounded = null;

            if (card.dataset.prompt) {
                promptInput.value = card.dataset.prompt;
            }
        });
    });

    // 3. Custom File Ingestion
    fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
        }
    });

    function handleFileUpload(file) {
        uploadedFile = file;
        currentPreset = null;
        presetsContainer.querySelectorAll('.target-card').forEach(c => c.classList.remove('active'));
        fileStatus.textContent = `${file.name} (${Math.round(file.size / 1024)} KB)`;

        const reader = new FileReader();
        reader.onload = (e) => {
            displayImg.src = e.target.result;
            analysisResults.raw = e.target.result;
            analysisResults.enhanced = null;
            analysisResults.grounded = null;
        };
        reader.readAsDataURL(file);
    }

    // 4. Directive Pills
    directiveChips.forEach(chip => {
        chip.addEventListener('click', () => {
            promptInput.value = chip.dataset.text;
        });
    });

    // 5. Run Analysis Trigger
    runBtn.addEventListener('click', async () => {
        const prompt = promptInput.value.trim();
        if (!prompt) {
            alert('Please enter an analysis directive.');
            return;
        }

        loader.classList.remove('hidden');

        try {
            const formData = new FormData();
            if (uploadedFile) {
                formData.append('image', uploadedFile);
            } else if (currentPreset) {
                formData.append('preset_filename', currentPreset);
            }
            formData.append('mission_prompt', prompt);
            formData.append('apply_enhancement', enhancementToggle.checked);

            const res = await fetch('/api/analyze', {
                method: 'POST',
                body: formData
            });

            if (!res.ok) {
                throw new Error(`Inference returned status code ${res.status}`);
            }

            const data = await res.json();
            if (data.success) {
                analysisResults.raw = `data:image/jpeg;base64,${data.raw_image_b64}`;
                analysisResults.enhanced = `data:image/jpeg;base64,${data.enhanced_image_b64}`;
                analysisResults.grounded = `data:image/jpeg;base64,${data.annotated_image_b64}`;
                analysisResults.heatmap = `data:image/jpeg;base64,${data.heatmap_image_b64}`;

                // Set grounded view active
                viewButtons.forEach(b => b.classList.remove('active'));
                const defaultViewBtn = document.querySelector('.control-btn[data-view="grounded"]');
                if (defaultViewBtn) defaultViewBtn.classList.add('active');
                displayImg.src = analysisResults.grounded;

                // Handle export dossier link
                const dossierLink = document.getElementById('dossier-download-link');
                if (dossierLink && data.report.dossier_export_url) {
                    dossierLink.href = data.report.dossier_export_url;
                    dossierLink.style.display = 'inline-flex';
                }

                renderReport(data.report);
            }
        } catch (err) {
            console.error('Execution error:', err);
            alert(`Analysis processing failed: ${err.message}`);
        } finally {
            loader.classList.add('hidden');
        }
    });

    // View Switching
    viewButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            viewButtons.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            const view = btn.dataset.view;
            if (view === 'grounded' && analysisResults.grounded) {
                displayImg.src = analysisResults.grounded;
            } else if (view === 'enhanced' && analysisResults.enhanced) {
                displayImg.src = analysisResults.enhanced;
            } else if (view === 'heatmap' && analysisResults.heatmap) {
                displayImg.src = analysisResults.heatmap;
            } else if (analysisResults.raw) {
                displayImg.src = analysisResults.raw;
            }
        });
    });

    function renderReport(report) {
        threatStatusEl.textContent = report.threat_status;
        threatStatusEl.style.backgroundColor = getStatusBg(report.threat_status);
        threatStatusEl.style.color = getStatusColor(report.threat_status);

        hazardScoreEl.textContent = `${report.hazard_severity_score} / 10.0`;
        narrativeEl.innerHTML = `<p>${report.natural_language_synthesis.replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</p>`;

        // Render Detections Table
        if (report.grounded_entities && report.grounded_entities.length > 0) {
            entitiesTbody.innerHTML = report.grounded_entities.map(e => `
                <tr>
                    <td style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--text-muted);">${e.id}</td>
                    <td style="font-weight: 600; color: var(--text-main);">${e.name}</td>
                    <td style="color: var(--text-body);">${e.spatial_region}</td>
                    <td style="font-family: var(--font-mono); font-weight: 600;">${e.confidence}</td>
                    <td>
                        <span style="display:inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; background: ${getBadgeBg(e.hazard_level)}; color: ${getBadgeTextColor(e.hazard_level)};">
                            ${e.hazard_level}
                        </span>
                    </td>
                </tr>
            `).join('');
        }

        protocolTextEl.textContent = report.strategic_recommendation;
        if (report.optical_telemetry && report.optical_telemetry.optical_cast_reduction_pct) {
            castPctEl.textContent = `${report.optical_telemetry.optical_cast_reduction_pct}%`;
        }
    }

    function getStatusBg(status) {
        if (status.includes('CRITICAL')) return '#FEE2E2';
        if (status.includes('MODERATE') || status.includes('WARNING')) return '#FEF3C7';
        return '#DCFCE7';
    }

    function getStatusColor(status) {
        if (status.includes('CRITICAL')) return '#DC2626';
        if (status.includes('MODERATE') || status.includes('WARNING')) return '#D97706';
        return '#16A34A';
    }

    function getBadgeBg(level) {
        if (level === 'CRITICAL') return '#FEE2E2';
        if (level === 'WARNING' || level === 'MODERATE') return '#FEF3C7';
        return '#F1F5F9';
    }

    function getBadgeTextColor(level) {
        if (level === 'CRITICAL') return '#DC2626';
        if (level === 'WARNING' || level === 'MODERATE') return '#B45309';
        return '#475569';
    }

    // 6. Optical Calibration Tuning
    const sliderRed = document.getElementById('slider-red');
    const sliderContrast = document.getElementById('slider-contrast');
    const sliderBlue = document.getElementById('slider-blue');
    const valRed = document.getElementById('val-red');
    const valContrast = document.getElementById('val-contrast');
    const valBlue = document.getElementById('val-blue');
    const tuneBtn = document.getElementById('tune-optical-btn');
    const labPreviewImg = document.getElementById('lab-preview-img');

    if (sliderRed) {
        sliderRed.addEventListener('input', () => valRed.textContent = `${sliderRed.value}x`);
        sliderContrast.addEventListener('input', () => valContrast.textContent = `${sliderContrast.value}x`);
        sliderBlue.addEventListener('input', () => valBlue.textContent = `${sliderBlue.value}x`);

        tuneBtn.addEventListener('click', async () => {
            try {
                const formData = new FormData();
                formData.append('preset_filename', currentPreset || 'ghost_net_reef.jpg');
                formData.append('red_gain', sliderRed.value);
                formData.append('contrast_factor', sliderContrast.value);
                formData.append('blue_damping', sliderBlue.value);

                const res = await fetch('/api/optical-enhance', {
                    method: 'POST',
                    body: formData
                });
                const data = await res.json();
                if (data.success && labPreviewImg) {
                    labPreviewImg.src = `data:image/jpeg;base64,${data.enhanced_b64}`;
                }
            } catch (err) {
                console.error('Calibration tuning error:', err);
            }
        });
    }

    // 7. Benchmark Validation Charts
    let chartsLoaded = false;
    async function loadBenchmarkCharts() {
        if (chartsLoaded) return;
        try {
            const res = await fetch('/api/benchmarks');
            const data = await res.json();
            if (!data.comparative_metrics) return;

            const models = data.comparative_metrics.map(m => m.model);
            const vqaAcc = data.comparative_metrics.map(m => m.vqa_accuracy);
            const bleu4 = data.comparative_metrics.map(m => m.bleu_4);
            const cider = data.comparative_metrics.map(m => m.cider);

            // Chart 1: VQA Accuracy
            const ctx1 = document.getElementById('chart-accuracy').getContext('2d');
            new Chart(ctx1, {
                type: 'bar',
                data: {
                    labels: models,
                    datasets: [{
                        label: 'VQA Top-1 Accuracy (%)',
                        data: vqaAcc,
                        backgroundColor: ['#94A3B8', '#64748B', '#0284C7', '#0F766E'],
                        borderRadius: 4
                    }]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, max: 100, grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } },
                        x: { grid: { display: false }, ticks: { color: '#334155', font: { weight: '500' } } }
                    },
                    plugins: { legend: { display: false } }
                }
            });

            // Chart 2: Language Metrics
            const ctx2 = document.getElementById('chart-metrics').getContext('2d');
            new Chart(ctx2, {
                type: 'bar',
                data: {
                    labels: models,
                    datasets: [
                        { label: 'CIDEr Score', data: cider, backgroundColor: '#0284C7', borderRadius: 4 },
                        { label: 'BLEU-4 Score', data: bleu4, backgroundColor: '#94A3B8', borderRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    scales: {
                        y: { beginAtZero: true, grid: { color: '#F1F5F9' }, ticks: { color: '#64748B' } },
                        x: { grid: { display: false }, ticks: { color: '#334155', font: { weight: '500' } } }
                    },
                    plugins: {
                        legend: { labels: { color: '#334155', font: { weight: '500' } } }
                    }
                }
            });

            // Populate Table
            const benchTbody = document.getElementById('bench-tbody');
            benchTbody.innerHTML = data.comparative_metrics.map(m => `
                <tr style="${m.model.includes('AquaSentinel') ? 'background-color: #F0FDF4; font-weight: 600;' : ''}">
                    <td>${m.model} <br><span style="font-size:0.7rem; color:#64748B; font-weight:400;">${m.type}</span></td>
                    <td style="color: #0284C7; font-weight:600;">${m.vqa_accuracy}%</td>
                    <td>${m.bleu_4}</td>
                    <td>${m.rouge_l}</td>
                    <td>${m.cider}</td>
                    <td style="color: #0F766E;">${m.grounding_recall_iou50}%</td>
                    <td style="font-size: 0.75rem; color:#475569;">${m.turbidity_robustness}</td>
                </tr>
            `).join('');

            chartsLoaded = true;
        } catch (err) {
            console.error('Failed to load benchmark data:', err);
        }
    }
});
