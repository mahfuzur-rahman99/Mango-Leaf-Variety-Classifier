// Get DOM elements
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const previewContainer = document.getElementById("preview-container");
const previewImage = document.getElementById("preview-image");
const predictionTitle = document.getElementById("prediction-title");
const predictionSubtitle = document.getElementById("prediction-subtitle");

// Open file dialog when button is clicked
browseBtn.addEventListener("click", () => fileInput.click());

// Handle file chosen via dialog
fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) {
        handleFile(file);
    }
});

// Drag events
["dragenter", "dragover"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        dropZone.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropZone.addEventListener(eventName, (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (eventName !== "drop") {
            dropZone.classList.remove("dragover");
        }
    });
});

// Drop file
dropZone.addEventListener("drop", (e) => {
    dropZone.classList.remove("dragover");
    const file = e.dataTransfer.files[0];
    if (file) {
        handleFile(file);
    }
});

// Also click entire drop zone to open file dialog
dropZone.addEventListener("click", () => fileInput.click());

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        setError("Please upload a valid image file (JPG, PNG, etc.)");
        return;
    }

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.classList.remove("hidden");
    };
    reader.readAsDataURL(file);

    // Send to backend
    uploadForPrediction(file);
}

function uploadForPrediction(file) {
    const formData = new FormData();
    formData.append("file", file); // must match app.py ("file")

    setLoading();

    fetch("/predict", {
        method: "POST",
        body: formData,
    })
    .then((res) => res.json())
    .then((data) => {
        if (data.error) {
            setError(data.error);
        } else {
            setSuccess(data.class_name, data.confidence);
        }
    })
    .catch((err) => {
        console.error(err);
        setError("Something went wrong. Please try again.");
    });
}

function setLoading() {
    predictionTitle.textContent = "Analyzing image...";
    predictionSubtitle.textContent = "Please wait while the model predicts the variety.";
}

function setSuccess(className, confidence) {
    predictionTitle.textContent = `Predicted: ${className}`;
    predictionSubtitle.textContent = `Confidence: ${confidence.toFixed(2)}%`;
}

function setError(message) {
    predictionTitle.textContent = "Error";
    predictionSubtitle.textContent = message;
}
