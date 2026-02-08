const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('fileInput');
const videoContainer = document.getElementById('video-container');
const processedFeed = document.getElementById('processed-feed');
const logList = document.getElementById('log-list');
const backBtn = document.getElementById('back-btn');
let statsTimer = null;

// Back Button
backBtn.addEventListener('click', () => {
    // 1. Stop polling
    if (statsTimer) clearInterval(statsTimer);

    // 2. Stop Feed (Important to stop backend processing if possible, or just kill the image updates)
    processedFeed.src = "";
    processedFeed.onload = null;
    processedFeed.onerror = null;

    // 3. Toggle Views
    videoContainer.classList.add('hidden');
    dropZone.classList.remove('hidden');

    // 4. Log
    addLog('System', 'Stopped analysis.', 'warning');

    // Reset inputs
    fileInput.value = '';
});

// Drag & Drop events
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#3b82f6';
});

dropZone.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255,255,255,0.1)';
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = 'rgba(255,255,255,0.1)';
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadVideo(files[0]);
    }
});

fileInput.addEventListener('change', (e) => {
    if (fileInput.files.length > 0) {
        uploadVideo(fileInput.files[0]);
    }
});

function uploadVideo(file) {
    const formData = new FormData();
    formData.append('video', file);

    addLog('System', `Uploading ${file.name}...`, 'info');

    fetch('/upload', {
        method: 'POST',
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if (data.filepath) {
                addLog('System', 'Upload successful. Starting analysis...', 'success');
                startProcessing(data.filepath);
            } else {
                addLog('Error', data.error, 'danger');
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addLog('Error', 'Upload failed.', 'danger');
        });
}

function startProcessing(filepath) {
    if (statsTimer) clearInterval(statsTimer);

    dropZone.classList.add('hidden');
    videoContainer.classList.remove('hidden');

    // Set the source of the image to the streaming endpoint
    // We add a timestamp to bypass cache
    processedFeed.src = `/video_feed?path=${encodeURIComponent(filepath)}&t=${new Date().getTime()}`;

    // Polling for Stats (Real-time updates)
    statsTimer = setInterval(() => {
        fetch('/stats')
            .then(response => response.json())
            .then(data => {
                document.getElementById('signal-count').innerText = data.signal;
                document.getElementById('helmet-count').innerText = data.helmet;
                document.getElementById('triple-count').innerText = data.triple;
                document.getElementById('traffic-helmet-count').innerText = data.traffic_helmet;
                document.getElementById('multiple-count').innerText = data.multiple;
            })
            .catch(err => console.error(err));
    }, 500);

    // Simulate finding violations (In a real app, you'd use Server-Sent Events or WebSockets)
    // Here we just listen for the image to load to confirm stream started
    processedFeed.onload = () => {
        addLog('System', 'Live stream active.', 'success');
    };

    processedFeed.onerror = () => {
        if (statsTimer) clearInterval(statsTimer);
        addLog('Error', 'Stream disconnected.', 'danger');
        setTimeout(() => {
            dropZone.classList.remove('hidden');
            videoContainer.classList.add('hidden');
        }, 2000);
    };
}

function addLog(source, message, type) {
    const item = document.createElement('div');
    item.className = `log-item ${type}`;
    item.innerHTML = `
        <span class="time">${new Date().toLocaleTimeString()} • ${source}</span>
        <span class="msg">${message}</span>
    `;
    logList.prepend(item);
}

// Navigation Logic
const navItems = {
    'nav-dashboard': 'section-dashboard',
    'nav-analytics': 'section-analytics',
    'nav-live': 'section-live',
    'nav-settings': 'section-settings'
};

Object.keys(navItems).forEach(navId => {
    const navLink = document.getElementById(navId);
    if (navLink) {
        navLink.addEventListener('click', (e) => {
            e.preventDefault();

            // 1. Update Active Link
            document.querySelectorAll('.nav-links li').forEach(li => li.classList.remove('active'));
            navLink.classList.add('active');

            // 2. Hide All Sections
            document.querySelectorAll('.content-section').forEach(section => {
                section.classList.add('hidden');
                section.style.display = 'none';
            });

            // 3. Show Target Section
            const targetId = navItems[navId];
            const targetSection = document.getElementById(targetId);
            if (targetSection) {
                targetSection.classList.remove('hidden');
                targetSection.style.display = 'block';
                // Remove display:none explicitly if it was set via style attribute
                if (targetSection.style.removeProperty) {
                    targetSection.style.removeProperty('display');
                }
                // Force block display for now to be safe, as my inline styles in HTML used display: none;
                targetSection.style.display = 'block';

                // Init Charts if switching to Analytics
                if (navId === 'nav-analytics') {
                    // Slight delay to ensure DOM is visible
                    setTimeout(initCharts, 50);
                }
            }
        });
    }
});

// Chart.js Initialization
let chartsInitialized = false;

function initCharts() {
    if (chartsInitialized) return;

    // Weekly Trends Chart (Line)
    const ctxWeekly = document.getElementById('weeklyChart');
    if (ctxWeekly) {
        new Chart(ctxWeekly.getContext('2d'), {
            type: 'line',
            data: {
                labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                datasets: [{
                    label: 'Violations This Week',
                    data: [12, 19, 3, 5, 2, 3, 10], // Mock Data
                    borderColor: '#3b82f6',
                    backgroundColor: 'rgba(59, 130, 246, 0.2)',
                    tension: 0.4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#94a3b8' } }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }


    // Violation Distribution (Pie)
    const ctxPie = document.getElementById('pieChart');
    if (ctxPie) {
        new Chart(ctxPie.getContext('2d'), {
            type: 'doughnut',
            data: {
                labels: ['Signal Jump', 'No Helmet', 'Triple Riding'],
                datasets: [{
                    data: [35, 45, 20], // Mock Data
                    backgroundColor: [
                        '#ef4444', // Red
                        '#f59e0b', // Orange
                        '#3b82f6'  // Blue
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { position: 'right', labels: { color: '#94a3b8' } }
                }
            }
        });
    }

    // Peak Hours (Bar) - instead of Heatmap
    const ctxBar = document.getElementById('peakHoursChart');
    if (ctxBar) {
        new Chart(ctxBar.getContext('2d'), {
            type: 'bar',
            data: {
                labels: ['18:00', '19:00', '20:00', '21:00', '22:00', '23:00', '00:00', '01:00'],
                datasets: [{
                    label: 'Violations Count',
                    data: [5, 12, 25, 30, 15, 8, 4, 2], // Mock Data
                    backgroundColor: '#8b5cf6',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    y: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    x: {
                        grid: { display: false },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
    }

    chartsInitialized = true;
}