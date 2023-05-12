const CAMERAS_MAP = ['Unity Video Capture', 'Unity Video Capture #2', 'Camera']

const videoContainer = document.getElementById('videos');

navigator.mediaDevices.enumerateDevices().then(devices => {
  const cameras = devices.filter(device => device.kind === 'videoinput');

  console.log(devices);

  cameras.forEach(camera => {
    const videoElement = document.createElement('video');

    videoElement.setAttribute('autoplay', '');
    videoElement.setAttribute('playsinline', '');
    videoElement.setAttribute('muted', '');
    videoElement.setAttribute('data-device-name', camera.label)
    videoElement.width = 320;
    videoElement.height = 240;

    if (!CAMERAS_MAP.includes(camera.label)) return

    navigator.mediaDevices.getUserMedia({
      video: {
        deviceId: camera.deviceId,
        width: { ideal: 320 },
        height: { ideal: 240 }
      }
    }).then(stream => {
      videoElement.srcObject = stream;
      videoContainer.appendChild(videoElement);
    });
  });
});