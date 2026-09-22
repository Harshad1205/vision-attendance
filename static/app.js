const video = document.getElementById('webcam');
const canvas = document.getElementById('snapshot');
const statusDiv = document.getElementById('status-bar');

if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: true })
        .then((stream) => { 
            video.srcObject = stream; 
        })
        .catch((err) => { 
            statusDiv.innerText = "Camera Error: " + err; 
            statusDiv.style.color = "#DA3633"; 
        });
}

function captureBlob() {
    return new Promise((resolve) => {
        canvas.width = video.videoWidth || 640;
        canvas.height = video.videoHeight || 480;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0);
        canvas.toBlob(resolve, 'image/jpeg');
    });
}

async function registerFace() {
    const nameInput = document.getElementById('reg-name');
    const name = nameInput.value.trim();
    if (!name) {
        alert("Enter a name first!");
        return;
    }

    statusDiv.innerText = "Capturing face for registration...";
    statusDiv.style.color = "#8957E5";

    const blob = await captureBlob();
    const data = new FormData();
    data.append("name", name);
    data.append("frame", blob, "face.jpg");

    try {
        const res = await fetch("/api/register", { method: "POST", body: data });
        const json = await res.json();
        statusDiv.innerText = json.message || "Registration complete.";
        statusDiv.style.color = res.ok ? "#2EA043" : "#DA3633";
        if (res.ok) {
            nameInput.value = "";
            refreshData();
        }
    } catch (err) {
        statusDiv.innerText = "Network Error: " + err;
        statusDiv.style.color = "#DA3633";
    }
}

async function submitScan(action) {
    statusDiv.innerText = `Analyzing face for ${action}...`;
    statusDiv.style.color = "#3B8ED0";

    const blob = await captureBlob();
    const data = new FormData();
    data.append("action", action);
    data.append("frame", blob, "scan.jpg");

    try {
        const res = await fetch("/api/scan", { method: "POST", body: data });
        const json = await res.json();
        if (res.ok) {
            statusDiv.innerText = `Matched: ${json.name} (${action})`;
            statusDiv.style.color = "#2EA043";
            refreshData();
        } else {
            statusDiv.innerText = json.message || "Face not recognized.";
            statusDiv.style.color = "#DA3633";
        }
    } catch (err) {
        statusDiv.innerText = "Scan Error: " + err;
        statusDiv.style.color = "#DA3633";
    }
}

async function refreshData() {
    try {
        const res = await fetch("/api/logs");
        const json = await res.json();
        document.getElementById("entry-val").innerText = json.latest_entry || "None";
        document.getElementById("exit-val").innerText = json.latest_exit || "None";
        const body = document.getElementById("logs-body");
        body.innerHTML = "";
        (json.logs || []).forEach((l) => {
            body.innerHTML += `<tr><td>${l.name}</td><td>${l.action}</td><td>${l.timestamp}</td></tr>`;
        });
    } catch (e) {
        console.error(e);
    }
}

setInterval(refreshData, 3000);
refreshData();