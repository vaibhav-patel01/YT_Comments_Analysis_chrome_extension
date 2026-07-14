document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyCj7Ri20R97uMNfvvdp4b-hIiufpNRJxj8';
  const SENTIMENT_API_URL = 'http://localhost:8000/predict'; // Your API endpoint for sentiment analysis

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;

    // Check if the URL is a valid YouTube URL
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.textContent = `YouTube Video ID: ${videoId}\nFetching comments...`;

      // Fetch comments from YouTube Data API (Passing API_KEY to fix scope)
      const comments = await fetchComments(videoId, API_KEY);
      
      if (comments.length === 0) {
        outputDiv.textContent += "\nNo comments found for this video.";
        return;
      }

      outputDiv.textContent += `\nFetched ${comments.length} comments. Sending for sentiment analysis...`;

      // Send comments to your API for sentiment prediction (Passing URL to fix scope)
      const predictions = await getSentimentPredictions(comments, SENTIMENT_API_URL);

      // Calculate and display sentiment distribution
      if (predictions) {
        const sentimentCounts = { "0": 0, "1": 0, "2": 0 };
        predictions.forEach(prediction => {
          if (sentimentCounts[prediction] !== undefined) {
            sentimentCounts[prediction]++;
          }
        });

        const total = predictions.length;
        const positivePercent = ((sentimentCounts["0"] / total) * 100).toFixed(2);
        const neutralPercent = ((sentimentCounts["1"] / total) * 100).toFixed(2);
        const negativePercent = ((sentimentCounts["2"] / total) * 100).toFixed(2);

        outputDiv.textContent += `\n\nSentiment Analysis Results:\nPositive: ${positivePercent}%\nNeutral: ${neutralPercent}%\nNegative: ${negativePercent}%`;
      }
    } else {
      outputDiv.textContent = "This is not a valid YouTube URL";
    }
  });
});

// Function to fetch all comments for the video
async function fetchComments(videoId, apiKey) {
  let comments = [];
  let pageToken = "";
  
  try {
    // Limit to 100 comments for rate limit purposes
    while (comments.length < 500) {
      const response = await fetch(`https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&maxResults=100&videoId=${videoId}&key=${apiKey}${pageToken ? `&pageToken=${pageToken}` : ''}`);
      const data = await response.json();
      
      if (data.items) {
        data.items.forEach(item => {
          comments.push(item.snippet.topLevelComment.snippet.textOriginal);
        });
      }

      pageToken = data.nextPageToken;
      if (!pageToken) break;
    }
  } catch (error) {
    console.error("Error fetching comments:", error);
  }

  return comments;
}

// Function to get sentiment predictions from your API
async function getSentimentPredictions(comments, apiUrl) {
  try {
    const response = await fetch(apiUrl, {
      method: "POST", // Change to "PUT" if aligning directly with your app.py backend
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ comments })
    });

    const result = await response.json();
    
    // Safely parse based on whether backend sends standard list or wrapped dict {"results": [...]}
    const dataList = result.results ? result.results : result;
    return dataList.map(item => item.sentiment); // Extract only sentiment values
    
  } catch (error) {
    console.error("Error fetching predictions:", error);
    const outputDiv = document.getElementById("output");
    outputDiv.textContent += "\nError fetching sentiment predictions.";
    return null;
  }
}
