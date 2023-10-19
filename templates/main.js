

const audioPlayer = document.getElementById('audioPlayer');






function togglePopup() {
    var popup = document.getElementById("popup");
    if (popup) {
        popup.style.display = (popup.style.display === "none" || popup.style.display === "") ? "block" : "none";
    } else {
        console.log("Popup element not found.");
    }








  
}









function updateAudioSource(event) {
    const file = event.target.files[0];
    const objectURL = URL.createObjectURL(file);

  
    audioPlayer.src = objectURL;

  
}

document.getElementById('file-upload').addEventListener('change', updateAudioSource);
