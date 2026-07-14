
const YT_API_KEY = 'AIzaSyCj7Ri20R97uMNfvvdp4b-hIiufpNRJxj8';
const SENTIMENT_API_URL = 'http://localhost:8000/predict'; // FastAPI backend
const MAX_COMMENTS = 500;
const TOP_N_DISPLAYED = 50;

const SENTIMENT_META = {
  0: { label: 'Negative', className: 'negative' },
  1: { label: 'Neutral', className: 'neutral' },
  2: { label: 'Positive', className: 'positive' }
};

const els = {
  videoIdLabel: document.getElementById('videoIdLabel'),
  states: {
    invalid: document.getElementById('state-invalid'),
    loading: document.getElementById('state-loading'),
    empty: document.getElementById('state-empty'),
    error: document.getElementById('state-error'),
    results: document.getElementById('state-results')
  },
  loadingTitle: document.getElementById('loadingTitle'),
  loadingBody: document.getElementById('loadingBody'),
  errorBody: document.getElementById('errorBody'),
  totalCount: document.getElementById('totalCount'),
  segPositive: document.getElementById('segPositive'),
  segNeutral: document.getElementById('segNeutral'),
  segNegative: document.getElementById('segNegative'),
  posPercent: document.getElementById('posPercent'),
  neuPercent: document.getElementById('neuPercent'),
  negPercent: document.getElementById('negPercent'),
  listCount: document.getElementById('listCount'),
  commentList: document.getElementById('commentList')
};

function showState(name) {
  Object.entries(els.states).forEach(([key, el]) => {
    el.classList.toggle('hidden', key !== name);
  });
}

function setLoading(title, body) {
  els.loadingTitle.textContent = title;
  els.loadingBody.textContent = body;
  showState('loading');
}

document.addEventListener('DOMContentLoaded', async () => {
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0]?.url || '';
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (!match || !match[1]) {
      showState('invalid');
      return;
    }

    const videoId = match[1];
    els.videoIdLabel.textContent = videoId;

    setLoading('Fetching comments…', 'Pulling the top comments from this video.');
    const comments = await fetchComments(videoId, YT_API_KEY);

    if (!comments || comments.length === 0) {
      showState('empty');
      return;
    }

    setLoading('Analyzing sentiment…', `Scoring ${comments.length} comments.`);
    const predictions = await getSentimentPredictions(comments, SENTIMENT_API_URL);

    if (!predictions) {
      showState('error');
      return;
    }

    renderResults(predictions);
  });
});

function renderResults(predictions) {
  const counts = { 0: 0, 1: 0, 2: 0 };
  predictions.forEach((item) => {
    const key = Number(item.sentiment);
    if (counts[key] !== undefined) counts[key]++;
  });

  const total = predictions.length;
  const pct = (n) => (total ? (n / total) * 100 : 0);
  const positivePct = pct(counts[2]);
  const neutralPct = pct(counts[1]);
  const negativePct = pct(counts[0]);

  els.totalCount.textContent = `${total.toLocaleString()} comment${total === 1 ? '' : 's'}`;

  // Animate the segmented bar in on the next frame.
  requestAnimationFrame(() => {
    els.segPositive.style.width = `${positivePct}%`;
    els.segNeutral.style.width = `${neutralPct}%`;
    els.segNegative.style.width = `${negativePct}%`;
  });

  els.posPercent.textContent = `${positivePct.toFixed(1)}%`;
  els.neuPercent.textContent = `${neutralPct.toFixed(1)}%`;
  els.negPercent.textContent = `${negativePct.toFixed(1)}%`;

  const shown = predictions.slice(0, TOP_N_DISPLAYED);
  els.listCount.textContent = `Showing ${shown.length} of ${total}`;
  els.commentList.innerHTML = '';

  shown.forEach((item) => {
    const meta = SENTIMENT_META[item.sentiment] || { label: 'Unknown', className: 'neutral' };

    const li = document.createElement('li');
    li.className = `comment-item comment-item--${meta.className}`;

    const text = document.createElement('p');
    text.className = 'comment-item__text';
    text.textContent = item.comment;

    const tag = document.createElement('span');
    tag.className = 'comment-item__tag';
    tag.textContent = meta.label;

    li.appendChild(text);
    li.appendChild(tag);
    els.commentList.appendChild(li);
  });

  showState('results');
}

// Fetch top-level comments for a video, paging until MAX_COMMENTS or the end.
async function fetchComments(videoId, apiKey) {
  let comments = [];
  let pageToken = '';

  try {
    while (comments.length < MAX_COMMENTS) {
      const response = await fetch(
        `https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId=${videoId}&key=${apiKey}${
          pageToken ? `&pageToken=${pageToken}` : ''
        }`
      );
      const data = await response.json();

      if (data.items) {
        data.items.forEach((item) => {
          comments.push(item.snippet.topLevelComment.snippet.textOriginal);
        });
      }

      pageToken = data.nextPageToken;
      if (!pageToken) break;
    }
  } catch (error) {
    console.error('Error fetching comments:', error);
  }

  return comments;
}

// Send comments to the local sentiment API and return per-comment predictions.
async function getSentimentPredictions(comments, apiUrl) {
  try {
    const response = await fetch(apiUrl, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ comments })
    });

    const result = await response.json();
    return result.results ? result.results : result;
  } catch (error) {
    console.error('Error fetching predictions:', error);
    return null;
  }
}