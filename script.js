const API_ENDPOINT = "http://127.0.0.1:8080/chat"; // 새 대화형 엔드포인트

const chat = document.getElementById('chat');
const input = document.getElementById('input');
const sendBtn = document.getElementById('send');
const chipsEl = document.getElementById('chips');

let sessionId = localStorage.getItem('ramen_session_id') || null;

const autoGrow = () => { input.style.height = 'auto'; input.style.height = Math.min(input.scrollHeight, 120) + 'px'; };
input.addEventListener('input', () => { sendBtn.disabled = !input.value.trim(); autoGrow(); });

chipsEl?.addEventListener('click', (e) => {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    input.value = chip.dataset.text;
    sendBtn.disabled = !input.value.trim();
    autoGrow();
    input.focus();
});

function addMsg(type, text){
    const row = document.createElement('div'); row.className = 'row ' + type;
    const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = type === 'user' ? '나' : '라면 소생자';
    const bubble = document.createElement('div'); bubble.className = 'bubble'; bubble.textContent = text;
    row.appendChild(avatar); row.appendChild(bubble); chat.appendChild(row);
    chat.parentElement.scrollTop = chat.parentElement.scrollHeight;
}

function addLoading(){
    const row = document.createElement('div'); row.className = 'row bot'; row.id = 'loadingRow';
    const avatar = document.createElement('div'); avatar.className = 'avatar'; avatar.textContent = '라면';
    const bubble = document.createElement('div'); bubble.className = 'bubble';
    const box = document.createElement('div'); box.className = 'loading'; box.innerHTML = '<div class="dot"></div><div class="dot"></div><div class="dot"></div>';
    bubble.appendChild(box); row.appendChild(avatar); row.appendChild(bubble); chat.appendChild(row);
    chat.parentElement.scrollTop = chat.parentElement.scrollHeight;
}
function removeLoading(){ document.getElementById('loadingRow')?.remove(); }

async function sendMessage(text){
    addMsg('user', text);
    input.value = ''; input.dispatchEvent(new Event('input'));
    addLoading();
    try {
        const res = await fetch(API_ENDPOINT, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text, session_id: sessionId })
        });
        const json = await res.json();
        removeLoading();
        if (!res.ok){ addMsg('bot', `❌ ${(json && json.detail) || '서버 오류'}`); return; }
        if (json.session_id && json.session_id !== sessionId){
            sessionId = json.session_id; localStorage.setItem('ramen_session_id', sessionId);
        }
        addMsg('bot', json.reply);
    } catch (e){
        removeLoading(); addMsg('bot', '❌ 서버에 연결하지 못했습니다.');
    }
}

sendBtn.addEventListener('click', ()=>{ const t = input.value.trim(); if (t) sendMessage(t); });
input.addEventListener('keydown', (e)=>{ if (e.key === 'Enter' && !e.shiftKey){ e.preventDefault(); const t = input.value.trim(); if (t) sendMessage(t); }});