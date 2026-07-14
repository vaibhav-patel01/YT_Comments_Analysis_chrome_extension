
document.addEventListener("DOMContentLoaded", async () => {
  const outputDiv = document.getElementById("output");
  const API_KEY = 'AIzaSyCj7Ri20R97uMNfvvdp4b-hIiufpNRJxj8';
  const SENTIMENT_API_URL = 'http://localhost:8000/predict'; // Port 8000 for FastAPI

  // Get the current tab's URL
  chrome.tabs.query({ active: true, currentWindow: true }, async (tabs) => {
    const url = tabs[0].url;

    // Check if the URL is a valid YouTube URL
    const youtubeRegex = /^https:\/\/(?:www\.)?youtube\.com\/watch\?v=([\w-]{11})/;
    const match = url.match(youtubeRegex);

    if (match && match[1]) {
      const videoId = match[1];
      outputDiv.textContent = `YouTube Video ID: ${videoId}\nFetching comments...`;

      // Fetch comments from YouTube Data API
      const comments = await fetchComments(videoId, API_KEY);
      
      if (comments.length === 0) {
        outputDiv.textContent += "\nNo comments found for this video.";
        return;
      }

      outputDiv.textContent += `\nFetched ${comments.length} comments. Sending for sentiment analysis...`;

      // Send comments to your API for sentiment prediction
      const predictions = await getSentimentPredictions(comments, SENTIMENT_API_URL);

      // Calculate and display sentiment distribution
      if (predictions) {
        const sentimentCounts = { "0": 0, "1": 0, "2": 0 };
        
        predictions.forEach(item => {
          const pred = String(item.sentiment);
          if (sentimentCounts[pred] !== undefined) {
            sentimentCounts[pred]++;
          }
        });

        const total = predictions.length;
        
        // Fixed mapping: 0 = Negative, 1 = Neutral, 2 = Positive
        const negativePercent = ((sentimentCounts["0"] / total) * 100).toFixed(2);
        const neutralPercent = ((sentimentCounts["1"] / total) * 100).toFixed(2);
        const positivePercent = ((sentimentCounts["2"] / total) * 100).toFixed(2);

        // Clear the loading text
        outputDiv.innerHTML = ""; 

        // Create the summary header
        const summaryDiv = document.createElement("div");
        summaryDiv.style.marginBottom = "15px";
        summaryDiv.innerHTML = `
          <strong>Sentiment Analysis Results:</strong><br/>
          🟢 Positive: ${positivePercent}%<br/>
          ⚪ Neutral: ${neutralPercent}%<br/>
          🔴 Negative: ${negativePercent}%<br/>
        `;
        outputDiv.appendChild(summaryDiv);

        // Create the Top 25 Comments section
        const listHeader = document.createElement("strong");
        listHeader.textContent = "Top 25 Comments:";
        outputDiv.appendChild(listHeader);

        const listContainer = document.createElement("ul");
        listContainer.style.paddingLeft = "0";
        listContainer.style.listStyleType = "none"; 
        
        // Map numbers to readable labels
        // Clean labels without emojis
        const sentimentLabels = {
          0: "Negative",
          1: "Neutral",
          2: "Positive"
        };

        // Slice the first 25 results
        const top25 = predictions.slice(0, 25);

        top25.forEach(item => {
          const li = document.createElement("li");
          li.style.marginBottom = "10px";
          li.style.paddingBottom = "10px";
          li.style.borderBottom = "1px solid #ddd";
          li.style.fontSize = "13px";
          
          const label = sentimentLabels[item.sentiment] || "Unknown";
          
          // Updated exact format: comment : sentiment
          li.textContent = `${item.comment} : ${label}`;
          
          listContainer.appendChild(li);
        });

        outputDiv.appendChild(listContainer);
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
    // Limit to 500 comments for rate limit purposes
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
      method: "POST", 
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ comments })
    });

    const result = await response.json();
    
    // Safely parse based on backend wrapper
    const dataList = result.results ? result.results : result;
    
    // RETURN FULL OBJECTS NOW, NOT JUST SENTIMENTS
    return dataList; 
    
  } catch (error) {
    console.error("Error fetching predictions:", error);
    const outputDiv = document.getElementById("output");
    outputDiv.textContent += "\nError fetching sentiment predictions.";
    return null;
  }
}