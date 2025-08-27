// static/script.js

const video = document.getElementById('video');
const outputImg = document.getElementById('output-img');
const currentSymbol = document.getElementById('current-symbol');
const word = document.getElementById('current-word');
const suggestions = [
  document.getElementById('suggest1'),
  document.getElementById('suggest2'),
  document.getElementById('suggest3'),
  document.getElementById('suggest4')
];
const speakBtn = document.getElementById('speak-btn');
const clearBtn = document.getElementById('clear-btn');
const addWordBtn = document.getElementById('add-word-btn');
const clearSentenceBtn = document.getElementById('clear-sentence-btn');
const sentenceElement = document.getElementById('sentence');

let currentSentence = '';

// Handle suggestion clicks
suggestions.forEach((el, i) => {
    el.addEventListener('click', () => {
        const suggestionText = el.querySelector('.suggestion-text');
        if (suggestionText && suggestionText.textContent.trim()) {
            addWordToSentence(suggestionText.textContent.trim());
            // Add visual feedback
            el.style.transform = 'scale(0.95)';
            setTimeout(() => {
                el.style.transform = '';
            }, 150);
        }
    });
});

function addWordToSentence(word) {
    if (currentSentence.trim() === '' || currentSentence === 'Start building your sentence...') {
        currentSentence = word;
    } else {
        currentSentence += ' ' + word;
    }
    sentenceElement.textContent = currentSentence;
    
    // Add visual feedback
    sentenceElement.style.color = '#2d3748';
    sentenceElement.parentElement.style.background = '#f0fff4';
    setTimeout(() => {
        sentenceElement.parentElement.style.background = '';
    }, 300);
}

speakBtn.addEventListener('click', () => {
    const textToSpeak = currentSentence && currentSentence !== 'Start building your sentence...' 
        ? currentSentence 
        : word.textContent;
    
    if (textToSpeak && textToSpeak.trim() !== '' && textToSpeak !== '-') {
        // Add visual feedback
        speakBtn.classList.add('loading');
        speakBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Speaking...';
        
        const utterance = new SpeechSynthesisUtterance(textToSpeak);
        utterance.onend = () => {
            speakBtn.classList.remove('loading');
            speakBtn.innerHTML = '<i class="fas fa-volume-up"></i> Speak Sentence';
        };
        speechSynthesis.speak(utterance);
    }
});

addWordBtn.addEventListener('click', () => {
    const currentWord = word.textContent.trim();
    if (currentWord && currentWord !== '-') {
        addWordToSentence(currentWord);
        // Add visual feedback
        addWordBtn.style.transform = 'scale(0.95)';
        setTimeout(() => {
            addWordBtn.style.transform = '';
        }, 150);
    }
});

clearSentenceBtn.addEventListener('click', () => {
    currentSentence = '';
    sentenceElement.textContent = 'Start building your sentence...';
    sentenceElement.style.color = '#a0aec0';
    
    // Add visual feedback
    clearSentenceBtn.style.transform = 'scale(0.95)';
    setTimeout(() => {
        clearSentenceBtn.style.transform = '';
    }, 150);
});

clearBtn.addEventListener('click', async () => {
    word.textContent = '-';
    currentSymbol.textContent = '-';
    suggestions.forEach(el => {
        const suggestionText = el.querySelector('.suggestion-text');
        if (suggestionText) {
            suggestionText.textContent = '';
        }
    });
    
    // Add visual feedback
    clearBtn.classList.add('loading');
    clearBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
    
    await fetch('/clear_word', { method: 'POST' });
    
    clearBtn.classList.remove('loading');
    clearBtn.innerHTML = '<i class="fas fa-eraser"></i> Clear All';
});

async function setupWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ 
      video: { 
        width: { ideal: 640 }, 
        height: { ideal: 480 },
        facingMode: 'user'
      } 
    });
    video.srcObject = stream;
    
    return new Promise(resolve => {
      video.onloadedmetadata = () => {
        // Update status indicator
        const statusIndicator = document.querySelector('.status-indicator');
        if (statusIndicator) {
          statusIndicator.innerHTML = '<div class="status-dot active"></div><span>Camera Active</span>';
        }
        resolve();
      };
    });
  } catch (error) {
    console.error('Error accessing webcam:', error);
    const statusIndicator = document.querySelector('.status-indicator');
    if (statusIndicator) {
      statusIndicator.innerHTML = '<div class="status-dot" style="background: #e53e3e;"></div><span>Camera Error</span>';
    }
  }
}

function captureFrame() {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas.toDataURL('image/jpeg');
}

async function sendFrame() {
  if (video.videoWidth === 0 || video.videoHeight === 0) return;
  const image = captureFrame();
  
  try {
    const response = await fetch('/process_frame', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image })
    });
    const data = await response.json();

    // Update UI with smooth transitions
    if (data.processed_image) {
      outputImg.src = data.processed_image;
    }
    
    // Update current symbol with animation
    const newSymbol = data.current_symbol || '-';
    if (currentSymbol.textContent !== newSymbol) {
      currentSymbol.style.transform = 'scale(1.1)';
      currentSymbol.textContent = newSymbol;
      setTimeout(() => {
        currentSymbol.style.transform = 'scale(1)';
      }, 200);
    }
    
    // Update current word with animation
    const newWord = data.word || '-';
    if (word.textContent !== newWord) {
      word.style.transform = 'scale(1.1)';
      word.textContent = newWord;
      setTimeout(() => {
        word.style.transform = 'scale(1)';
      }, 200);
    }
    
    // Update suggestions
    suggestions.forEach((el, i) => {
      const suggestionText = el.querySelector('.suggestion-text');
      if (suggestionText) {
        const newSuggestion = data['word' + (i + 1)] || '';
        suggestionText.textContent = newSuggestion;
        
        // Show/hide suggestion buttons based on content
        if (newSuggestion) {
          el.style.opacity = '1';
          el.style.pointerEvents = 'auto';
        } else {
          el.style.opacity = '0.3';
          el.style.pointerEvents = 'none';
        }
      }
    });
  } catch (error) {
    console.error('Error processing frame:', error);
  }
}

async function loop() {
  await sendFrame();
  setTimeout(loop, 200); // Adjust interval as needed
}

// Initialize the application
document.addEventListener('DOMContentLoaded', () => {
  // Set initial sentence text
  sentenceElement.textContent = 'Start building your sentence...';
  sentenceElement.style.color = '#a0aec0';
  
  // Initialize suggestion buttons
  suggestions.forEach(el => {
    const suggestionText = el.querySelector('.suggestion-text');
    if (suggestionText) {
      suggestionText.textContent = '';
    }
    el.style.opacity = '0.3';
    el.style.pointerEvents = 'none';
  });
  
  // Start webcam and main loop
  setupWebcam().then(() => {
    setTimeout(loop, 1000); // Wait a bit before starting
  }).catch(error => {
    console.error('Failed to initialize webcam:', error);
  });
});
