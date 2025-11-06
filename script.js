const fileInput = document.getElementById('fileInput');
const processBtn = document.getElementById('processBtn');
const chunksList = document.getElementById('chunksList');
const statsArea = document.getElementById('statsArea');
const chunkSizeInput = document.getElementById('chunkSize');
const askBtn = document.getElementById('askBtn');
const questionInput = document.getElementById('question');
const retrievedDiv = document.getElementById('retrieved');
const answerDiv = document.getElementById('answer');
const backendUrlInput = document.getElementById('backendUrl');
const clearBtn = document.getElementById('clearBtn');

let docs = [], chunks = [], vocabulary = new Set(), idf = {};

// --- PDF Extraction ---
async function extractTextFromPDF(file) {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const pages = [];

  for (let p = 1; p <= pdf.numPages; p++) {
    try {
      const page = await pdf.getPage(p);
      const content = await page.getTextContent();
      const text = content.items.map(i => i.str).join(' ').replace(/\s+/g, ' ').trim();
      pages.push({ pageNum: p, text });
    } catch (e) {
      console.error('Error reading page', p, e);
      pages.push({ pageNum: p, text: '' });
    }
  }
  return pages;
}

// --- Helpers ---
function splitToChunks(text, wordsPerChunk) {
  const words = text.split(/\s+/).filter(Boolean);
  const chunks = [];
  for (let i = 0; i < words.length; i += wordsPerChunk)
    chunks.push(words.slice(i, i + wordsPerChunk).join(' '));
  return chunks;
}

function tokenize(s) {
  return s.toLowerCase().replace(/[^a-z0-9\s]/g, '').split(/\s+/).filter(Boolean);
}

function buildIndex() {
  vocabulary = new Set();
  chunks.forEach(c => {
    const terms = tokenize(c.text);
    c.tf = {};
    terms.forEach(t => { vocabulary.add(t); c.tf[t] = (c.tf[t] || 0) + 1; });
    c.tokens = terms.length;
  });

  const N = chunks.length;
  idf = {};
  vocabulary.forEach(t => {
    let df = 0; chunks.forEach(c => { if (c.tf[t]) df++; });
    idf[t] = Math.log((N + 1) / (df + 1)) + 1;
  });

  chunks.forEach(c => {
    c.vector = {}; let sumsq = 0;
    Object.entries(c.tf).forEach(([t, tfv]) => {
      const val = (tfv / c.tokens) * idf[t];
      c.vector[t] = val; sumsq += val * val;
    });
    c.norm = Math.sqrt(sumsq) || 1e-9;
  });
}

function queryIndex(q) {
  const qterms = tokenize(q);
  const qtf = {};
  qterms.forEach(t => qtf[t] = (qtf[t] || 0) + 1);
  const qvec = {}; let sumsq = 0;
  Object.entries(qtf).forEach(([t, tfv]) => {
    const idfv = idf[t] || Math.log((chunks.length + 1) / 1) + 1;
    const val = (tfv / qterms.length) * idfv;
    qvec[t] = val; sumsq += val * val;
  });
  const qnorm = Math.sqrt(sumsq) || 1e-9;

  return chunks.map(c => {
    let dot = 0;
    Object.entries(qvec).forEach(([t, v]) => { if (c.vector[t]) dot += v * c.vector[t]; });
    const sim = dot / (qnorm * c.norm);
    return { ...c, score: sim };
  }).sort((a, b) => b.score - a.score).slice(0, 6);
}

function renderChunksList() {
  chunksList.innerHTML = '';
  chunks.forEach(c => {
    const el = document.createElement('div');
    el.className = 'chunk';
    el.innerHTML = `<strong>Page ${c.page}</strong> — ${c.text.slice(0, 120)}${c.text.length > 120 ? '...' : ''}`;
    chunksList.appendChild(el);
  });
}

// --- Event Handlers ---
processBtn.addEventListener('click', async () => {
  const files = Array.from(fileInput.files);
  if (!files.length) return alert('Select PDF files first.');
  docs = []; chunks = [];
  statsArea.textContent = 'Extracting text...';

  for (const f of files) {
    const pages = await extractTextFromPDF(f);
    const fullText = pages.map(p => p.text).join('\n');
    docs.push({ name: f.name, pages, fullText });

    for (const p of pages) {
      if (!p.text || p.text.length < 30) continue;
      const cks = splitToChunks(p.text, parseInt(chunkSizeInput.value) || 400);
      cks.forEach(ck => chunks.push({
        id: 'c' + (chunks.length + 1),
        text: ck,
        sourceName: f.name,
        page: p.pageNum
      }));
    }
  }

  buildIndex();
  renderChunksList();
  statsArea.textContent = `${docs.length} doc(s), ${chunks.length} chunk(s), vocab ${vocabulary.size}.`;
});

askBtn.addEventListener('click', async () => {
  const q = questionInput.value.trim();
  if (!q) return alert('Enter a question.');
  if (!chunks.length) return alert('Build the index first.');
  retrievedDiv.innerHTML = '';
  answerDiv.textContent = 'Searching...';

  const top = queryIndex(q);
  top.forEach((t, i) => {
    const el = document.createElement('div');
    el.innerHTML = `<div><strong>Rank ${i + 1}</strong> — Page ${t.page} (${t.score.toFixed(3)})</div><div class="context">${t.text}</div>`;
    retrievedDiv.appendChild(el);
  });

  const backend = backendUrlInput.value.trim();
  if (backend) {
    try {
      const resp = await fetch(backend, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, contexts: top.map(t => ({ text: t.text, page: t.page, score: t.score })) })
      });
      const data = await resp.json();
      answerDiv.textContent = data.answer || JSON.stringify(data);
      return;
    } catch {
      answerDiv.textContent = 'Backend error — showing demo summary.';
    }
  }

  const combined = top.slice(0, 3).map(t => t.text).join(' ');
  answerDiv.textContent = combined.split(/\s+/).slice(0, 300).join(' ') + ' ...';
});

clearBtn.addEventListener('click', () => {
  fileInput.value = '';
  docs = []; chunks = []; vocabulary = new Set(); idf = {};
  chunksList.innerHTML = '';
  statsArea.textContent = 'Cleared.';
  retrievedDiv.innerHTML = '';
  answerDiv.textContent = 'No answer yet — ask a question.';
});
