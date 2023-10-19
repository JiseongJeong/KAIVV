

const audio = document.getElementById('audio');
const audioSource = document.getElementById('audio-source');

const filename = "{{ audio_file.split('/')[-1] }}";
const relativePath = "static/audio/" + filename;
audioSource.src = relativePath;
audio.load();

function playAudio() {
    audio.play();
}

audio.addEventListener('canplaythrough', function() {
  
    playAudio();
});

audio.addEventListener('ended', function() {
    audio.removeEventListener('ended', onAudioEnded);
    audio.pause();
    audio.currentTime = 0;
});

const scripts = {{ scripts | tojson | safe }};
const scriptDisplay = document.getElementById('scriptDisplay');
let index = 0;

setTimeout(displayNextScript, 1000);

function displayNextScript() {
    if (index < scripts.length) {
        scriptDisplay.innerText = scripts[index];
        index++;
        setTimeout(displayNextScript, 2000); 
    }
}
