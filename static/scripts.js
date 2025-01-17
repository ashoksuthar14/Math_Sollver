function captureFromCamera() {
    const video = document.getElementById("video");
    const canvas = document.getElementById("canvas");
    const cameraForm = document.getElementById("camera-form");
    const cameraInput = document.getElementById("camera-image");

    navigator.mediaDevices.getUserMedia({ video: true }).then((stream) => {
        video.srcObject = stream;
    });

    video.addEventListener("click", () => {
        const context = canvas.getContext("2d");
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        const imageData = canvas.toDataURL("image/png");
        cameraInput.value = imageData;
        cameraForm.submit();
    });
}
