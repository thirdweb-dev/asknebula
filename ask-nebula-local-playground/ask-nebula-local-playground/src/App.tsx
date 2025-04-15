import { useState, useEffect } from "react";
import "./App.css";

interface Post {
  post: string;
}

interface Tweet {
  post: string;
}

// Different possible content structures
interface SinglePostContent {
  post: string;
}

interface ThreadContent {
  tweets?: Post[];
  tweet1?: string;
  tweet2?: string;
  tweet3?: string;
}

// Union type for all possible content structures
type ContentType = SinglePostContent | ThreadContent | Post[];

interface ThreadResponse {
  content: ContentType;
  type: string;
  thread_model?: {
    tweet1: string;
    tweet2: string;
    tweet3: string;
  };
}

// Interface for history items
interface HistoryItem {
  id: string;
  query: string;
  timestamp: Date;
  responseTime: number; // Time in milliseconds
  response: ThreadResponse | null;
}

function App() {
  const [query, setQuery] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [responses, setResponses] = useState<ThreadResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [activeHistoryId, setActiveHistoryId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [responseTime, setResponseTime] = useState<number | null>(null);

  // Function to clean and deduplicate URLs in text
  const cleanText = (text: string): string => {
    if (!text) return "";

    // Check if text has duplicate URLs
    const lines = text.split("\n");
    const cleanedLines = [];
    const seenUrls = new Set();
    const urlRegex = /^(https?:\/\/[^\s]+)$/;
    const introTextRegex =
      /check\s+(?:it\s+)?out\s+(?:here|at)?:?|here\s+(?:is|are)\s+(?:the|some)?\s+links?:?|(?:link|url)s?(?:\s+(?:is|are))?:?/i;

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim();

      // Skip empty lines
      if (!line) {
        cleanedLines.push("");
        continue;
      }

      // Check if line is a URL
      const urlMatch = line.match(urlRegex);
      if (urlMatch) {
        const url = urlMatch[1];
        if (seenUrls.has(url)) {
          // Skip duplicate URL
          continue;
        }
        seenUrls.add(url);
        cleanedLines.push(line);
        continue;
      }

      // Check if this line introduces URLs
      if (introTextRegex.test(line)) {
        // Check if next line is a URL we've already seen or will see
        const nextIndex = i + 1;
        if (nextIndex < lines.length) {
          const nextLine = lines[nextIndex].trim();
          const nextUrlMatch = nextLine.match(urlRegex);

          if (nextUrlMatch) {
            const nextUrl = nextUrlMatch[1];
            // If this is introducing a URL that appears elsewhere, skip this intro line
            if (
              seenUrls.has(nextUrl) ||
              lines.filter((l) => l.includes(nextUrl)).length > 1
            ) {
              continue;
            }
          }
        }
      }

      cleanedLines.push(line);
    }

    return cleanedLines.join("\n");
  };

  // Function to format text with clickable links
  const formatTextWithLinks = (text: string): React.ReactNode => {
    if (!text) return null;

    // First clean the text to remove duplicate URLs
    const cleanedText = cleanText(text);

    // Regex to match URLs more precisely
    const urlRegex = /(https?:\/\/[^\s<]+)/g;

    // If there are no URLs, return the text as-is
    if (!cleanedText.match(urlRegex)) {
      return cleanedText;
    }

    // More direct approach: Find all URLs and only keep the first occurrence of each
    const seen = new Set<string>();
    let uniqueUrlsText = cleanedText;

    // Find all URLs in the text
    const matches = Array.from(cleanedText.matchAll(urlRegex));

    // Process URLs from end to beginning to avoid position shifts
    for (let i = matches.length - 1; i >= 0; i--) {
      const match = matches[i];
      const url = match[0];
      const startPos = match.index;

      if (seen.has(url)) {
        // Remove this duplicate URL from the text
        uniqueUrlsText =
          uniqueUrlsText.substring(0, startPos) +
          uniqueUrlsText.substring(startPos + url.length);
      } else {
        seen.add(url);
      }
    }

    // Now create placeholders for the remaining (unique) URLs
    const uniqueUrls = Array.from(seen);
    let processedText = uniqueUrlsText;
    const placeholders: Record<string, string> = {};

    uniqueUrls.forEach((url, index) => {
      const placeholder = `__URL_PLACEHOLDER_${index}__`;
      placeholders[placeholder] = url;

      // Escape the URL for regex replacement
      const urlEscaped = url.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

      // Replace with placeholder
      processedText = processedText.replace(
        new RegExp(urlEscaped),
        placeholder
      );
    });

    // Split by placeholders and rebuild with React elements
    const parts = processedText.split(/((__URL_PLACEHOLDER_\d+__))/);

    const result = parts.map((part, i) => {
      // Check if this part is a placeholder
      if (part.startsWith("__URL_PLACEHOLDER_")) {
        const url = placeholders[part];
        if (url) {
          // Check if this URL is the only content in this paragraph/line
          const isIsolatedLink = processedText.trim() === part.trim();

          return (
            <a
              key={i}
              href={url}
              target="_blank"
              rel="noopener noreferrer"
              className={`blockchain-link ${
                isIsolatedLink ? "isolated-link" : ""
              }`}
            >
              {url.length > 60
                ? `${url.substring(0, 30)}...${url.substring(url.length - 25)}`
                : url}
            </a>
          );
        }
      }
      return part;
    });

    return result;
  };

  // Function to format response time
  const formatResponseTime = (ms: number): string => {
    if (ms < 1000) {
      return `${ms.toFixed(0)}ms`;
    } else {
      return `${(ms / 1000).toFixed(1)}s`;
    }
  };

  // Function to generate a unique ID
  const generateId = (): string => {
    return Date.now().toString(36) + Math.random().toString(36).substring(2);
  };

  // Function to add a query to history
  const addToHistory = (
    queryText: string,
    responseData: ThreadResponse | null,
    responseDuration: number
  ) => {
    const newHistoryItem: HistoryItem = {
      id: generateId(),
      query: queryText,
      timestamp: new Date(),
      responseTime: responseDuration,
      response: responseData,
    };

    setHistory((prevHistory) => [newHistoryItem, ...prevHistory]);
    setActiveHistoryId(newHistoryItem.id);
  };

  // Function to load a history item
  const loadHistoryItem = (id: string) => {
    const item = history.find((h) => h.id === id);
    if (item) {
      setActiveHistoryId(id);
      setResponses(item.response);
      setQuery(item.query);
      setResponseTime(item.responseTime);
      setError(null);
    }
  };

  // Function to format timestamp
  const formatTimestamp = (date: Date): string => {
    return new Date(date).toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
    });
  };

  // Try handling the Twitter thread response by sending a simplified query
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setError(null);
    setResponses(null);
    setResponseTime(null);
    setActiveHistoryId(null);

    const startTime = performance.now();

    try {
      // First attempt with a modified request that should help avoid serialization issues
      const response = await fetch("http://localhost:5001/api/create-thread", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Accept: "application/json",
          "X-Simplified-Response": "true", // Signal the server to simplify the response if it has middleware to do so
        },
        body: JSON.stringify({
          query: query.trim(),
          simplified: true, // Additional flag that could be used by server
        }),
      });

      // First try to get the raw text
      const rawText = await response.text();

      // Calculate response time
      const endTime = performance.now();
      const duration = endTime - startTime;
      setResponseTime(duration);

      // Try to parse the response as JSON
      let data;
      try {
        data = JSON.parse(rawText);
      } catch (parseErr) {
        // If the error contains "TwitterThread is not JSON serializable"
        if (rawText.includes("TwitterThread is not JSON serializable")) {
          // Create a simulated response with a fallback message
          const fallbackResponse = {
            type: "post",
            content: {
              post: "Your query was processed, but there was a technical issue displaying the full response. Please try again with a different query or contact the administrator to fix the server serialization issue.",
            },
          };

          setResponses(fallbackResponse);
          // Add to history even with the error
          addToHistory(query.trim(), fallbackResponse, duration);

          throw new Error(
            "Server unable to serialize TwitterThread response. The query likely worked, but the response couldn't be sent properly."
          );
        } else {
          throw new Error(
            `Failed to parse server response as JSON: ${
              parseErr instanceof Error ? parseErr.message : "Unknown error"
            }`
          );
        }
      }

      if (!response.ok) {
        throw new Error(data.error || `Server error: ${response.status}`);
      }

      setResponses(data);
      // Add successful query to history
      addToHistory(query.trim(), data, duration);

      // Clear the input field after successful response
      setQuery("");
    } catch (err) {
      const endTime = performance.now();
      const duration = endTime - startTime;
      setResponseTime(duration);

      setError(
        err instanceof Error ? err.message : "An unknown error occurred"
      );
    } finally {
      setIsLoading(false);
    }
  };

  // Function to render content based on response structure
  const renderContent = () => {
    if (!responses || !responses.content) {
      return <p className="no-responses">No content in response</p>;
    }

    const content = responses.content;

    // Case 1: Thread model direct access if available
    if (responses.thread_model) {
      const tweets = [
        responses.thread_model.tweet1,
        responses.thread_model.tweet2,
        responses.thread_model.tweet3,
      ];

      // Check for duplicate links across tweets
      const processedTweets = deduplicateLinksAcrossTweets(tweets);

      return (
        <div className="response-cards thread">
          <div className="response-card">
            <h3>Tweet 1</h3>
            {renderTextWithLinks(processedTweets[0])}
          </div>
          <div className="response-card">
            <h3>Tweet 2</h3>
            {renderTextWithLinks(processedTweets[1])}
          </div>
          <div className="response-card">
            <h3>Tweet 3</h3>
            {renderTextWithLinks(processedTweets[2])}
          </div>
        </div>
      );
    }

    // Case 2: Content with tweet1, tweet2, tweet3 format
    if ("tweet1" in content && "tweet2" in content && "tweet3" in content) {
      const tweets = [
        content.tweet1 as string,
        content.tweet2 as string,
        content.tweet3 as string,
      ];

      // Check for duplicate links across tweets
      const processedTweets = deduplicateLinksAcrossTweets(tweets);

      return (
        <div className="response-cards thread">
          <div className="response-card">
            <h3>Tweet 1</h3>
            {renderTextWithLinks(processedTweets[0])}
          </div>
          <div className="response-card">
            <h3>Tweet 2</h3>
            {renderTextWithLinks(processedTweets[1])}
          </div>
          <div className="response-card">
            <h3>Tweet 3</h3>
            {renderTextWithLinks(processedTweets[2])}
          </div>
        </div>
      );
    }

    // Case 3: Content is an array of posts
    if (Array.isArray(content)) {
      const posts = content.map((item) =>
        typeof item === "string" ? item : item.post
      );

      // Check for duplicate links across posts
      const processedPosts = deduplicateLinksAcrossTweets(posts);

      return (
        <div className="response-cards">
          {processedPosts.map((post, index) => (
            <div className="response-card" key={index}>
              {renderTextWithLinks(post)}
            </div>
          ))}
        </div>
      );
    }

    // Case 4: Content has tweets array (TwitterThread)
    if ("tweets" in content && Array.isArray(content.tweets)) {
      const posts = content.tweets.map((tweet) => tweet.post);

      // Check for duplicate links across tweets
      const processedPosts = deduplicateLinksAcrossTweets(posts);

      return (
        <div className="response-cards">
          {processedPosts.map((post, index) => (
            <div className="response-card" key={index}>
              {renderTextWithLinks(post)}
            </div>
          ))}
        </div>
      );
    }

    // Case 5: Content has a single post
    if ("post" in content) {
      return (
        <div className="response-cards">
          <div className="response-card">
            {renderTextWithLinks(content.post as string)}
          </div>
        </div>
      );
    }

    // Fallback: display raw content for debugging
    return (
      <div className="response-card">
        <p className="debug-info">Response structure unknown.</p>
        <pre>{JSON.stringify(content, null, 2)}</pre>
      </div>
    );
  };

  // Special direct render function to ensure no duplicate links
  const renderTextWithLinks = (text: string) => {
    if (!text) return null;

    // Extract URLs
    const urlRegex = /(https?:\/\/[^\s<]+)/g;
    const urls = [];
    let match;

    // Find all URLs in the text
    while ((match = urlRegex.exec(text)) !== null) {
      urls.push({
        url: match[0],
        index: match.index,
      });
    }

    // If no URLs, just return the text
    if (urls.length === 0) {
      return <p>{text}</p>;
    }

    // De-duplicate URLs by keeping only the first occurrence of each
    const uniqueUrls = [];
    const seenUrls = new Set();

    for (const item of urls) {
      if (!seenUrls.has(item.url)) {
        seenUrls.add(item.url);
        uniqueUrls.push(item);
      }
    }

    // Sort by position in text
    uniqueUrls.sort((a, b) => a.index - b.index);

    // Check if the text is basically just a URL
    const trimmedText = text.trim();
    const isJustUrl =
      uniqueUrls.length === 1 && trimmedText === uniqueUrls[0].url;

    // For text with intro + URL pattern, preserve both but style the URL specially
    const hasIntroText =
      uniqueUrls.length === 1 &&
      (trimmedText.includes("Check out") ||
        trimmedText.includes("view") ||
        trimmedText.includes("here:") ||
        trimmedText.includes("details") ||
        trimmedText.includes("transaction"));

    // If it's just a URL, render as isolated link
    if (isJustUrl) {
      const url = uniqueUrls[0].url;
      return (
        <p>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="blockchain-link isolated-link"
          >
            {url.length > 60
              ? `${url.substring(0, 30)}...${url.substring(url.length - 25)}`
              : url}
          </a>
        </p>
      );
    }

    // If it has intro text + URL, preserve intro but highlight the link
    if (hasIntroText) {
      // Build the elements with special treatment for the URL
      let lastIndex = 0;
      const elements = [];
      const url = uniqueUrls[0].url;
      const urlIndex = uniqueUrls[0].index;

      // Add text before the URL
      if (urlIndex > 0) {
        elements.push(
          <span key="intro-text">{text.substring(0, urlIndex)}</span>
        );
      }

      // Add the URL as a special link
      elements.push(
        <div key="link-container" className="isolated-link-container">
          <a
            key="special-url"
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="blockchain-link isolated-link"
          >
            {url.length > 60
              ? `${url.substring(0, 30)}...${url.substring(url.length - 25)}`
              : url}
          </a>
        </div>
      );

      return <p>{elements}</p>;
    }

    // Otherwise, build the text with URLs replaced by links
    let lastIndex = 0;
    const elements = [];

    uniqueUrls.forEach((item, i) => {
      // Add text before this URL
      if (item.index > lastIndex) {
        elements.push(
          <span key={`text-${i}`}>{text.substring(lastIndex, item.index)}</span>
        );
      }

      // Add the URL as a link
      elements.push(
        <a
          key={`url-${i}`}
          href={item.url}
          target="_blank"
          rel="noopener noreferrer"
          className="blockchain-link"
        >
          {item.url.length > 60
            ? `${item.url.substring(0, 30)}...${item.url.substring(
                item.url.length - 25
              )}`
            : item.url}
        </a>
      );

      lastIndex = item.index + item.url.length;
    });

    // Add any remaining text after the last URL
    if (lastIndex < text.length) {
      elements.push(<span key={`text-last`}>{text.substring(lastIndex)}</span>);
    }

    return <p>{elements}</p>;
  };

  // Function to deduplicate links across multiple tweets/posts
  const deduplicateLinksAcrossTweets = (texts: string[]): string[] => {
    const urlRegex = /(https?:\/\/[^\s<]+)/g;
    const seenUrls = new Set<string>();
    const processedTexts: string[] = [];

    // First pass: collect all URLs
    texts.forEach((text) => {
      const matches = text?.match(urlRegex) || [];
      matches.forEach((url) => seenUrls.add(url));
    });

    // Second pass: process each text to remove duplicate URLs
    texts.forEach((text) => {
      if (!text) {
        processedTexts.push("");
        return;
      }

      // Process text to handle URLs in paragraphs or sentences
      // This is a more thorough approach that works for URLs in running text
      const urlsInThisText = new Map<string, number>(); // URL -> first occurrence position
      let processedText = text;

      // Find all URLs and their positions in this text
      let match;
      while ((match = urlRegex.exec(text)) !== null) {
        const url = match[0];
        const position = match.index;

        // Record only the first occurrence of each URL
        if (!urlsInThisText.has(url)) {
          urlsInThisText.set(url, position);
        }
      }

      // Reset regex state
      urlRegex.lastIndex = 0;

      // Process from end to beginning to avoid position shifts
      const sortedUrls = Array.from(urlsInThisText.entries()).sort(
        (a, b) => b[1] - a[1]
      ); // Sort by position, descending

      for (const [url, pos] of sortedUrls) {
        const urlEscaped = url.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

        // Keep track of whether we've seen this first occurrence
        let foundFirst = false;

        // Create a new string with all occurrences removed except the first one
        let newText = "";
        let lastIndex = 0;
        let urlIndex;

        // Custom replacement to keep only first occurrence
        while ((urlIndex = processedText.indexOf(url, lastIndex)) !== -1) {
          if (!foundFirst) {
            // Keep the first occurrence
            newText += processedText.substring(
              lastIndex,
              urlIndex + url.length
            );
            foundFirst = true;
          } else {
            // Skip other occurrences
            newText += processedText.substring(lastIndex, urlIndex);
          }
          lastIndex = urlIndex + url.length;
        }

        // Add the rest of the text
        newText += processedText.substring(lastIndex);
        processedText = newText;
      }

      // Process text by lines to remove intro text for duplicate URLs
      const lines = processedText.split("\n");
      const cleanedLines = [];

      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim();

        // Skip empty lines but preserve them
        if (!line) {
          cleanedLines.push("");
          continue;
        }

        // Check if line is introducing links
        if (
          line.match(
            /check\s+(?:it\s+)?out\s+(?:here|at)?:?|here\s+(?:is|are)\s+(?:the|some)?\s+links?:?|(?:link|url)s?(?:\s+(?:is|are))?:?/i
          )
        ) {
          // Check if next line is a URL
          if (i + 1 < lines.length) {
            const nextLine = lines[i + 1].trim();
            if (nextLine.match(/^(https?:\/\/[^\s]+)$/)) {
              // If this URL appears in multiple tweets, skip the intro line
              if (texts.filter((t) => t && t.includes(nextLine)).length > 1) {
                continue;
              }
            }
          }
        }

        cleanedLines.push(line);
      }

      processedTexts.push(cleanedLines.join("\n"));
    });

    return processedTexts;
  };

  // Function to toggle history panel
  const toggleHistory = () => {
    setSidebarOpen(!sidebarOpen);
  };

  return (
    <div className="app-container">
      {/* History Panel Toggle Button */}
      <button
        className="history-toggle"
        onClick={toggleHistory}
        aria-label={sidebarOpen ? "Close history" : "Open history"}
      >
        <div className="history-toggle-icon">
          <span className="history-dot"></span>
          <span className="history-dot"></span>
          <span className="history-dot"></span>
        </div>
        <span className="history-label">History</span>
      </button>

      {/* History Panel */}
      <div className={`history-panel ${sidebarOpen ? "open" : ""}`}>
        <div className="history-panel-header">
          <h2>Query History</h2>
          <button className="close-history" onClick={toggleHistory}>
            ×
          </button>
        </div>
        <div className="history-panel-content">
          {history.length === 0 ? (
            <div className="history-empty">No queries yet</div>
          ) : (
            <ul className="history-list">
              {history.map((item) => (
                <li
                  key={item.id}
                  className={`history-item ${
                    activeHistoryId === item.id ? "active" : ""
                  }`}
                  onClick={() => loadHistoryItem(item.id)}
                >
                  <div className="history-query">
                    {item.query.length > 40
                      ? item.query.substring(0, 37) + "..."
                      : item.query}
                  </div>
                  <div className="history-meta">
                    <span className="history-time">
                      {formatTimestamp(item.timestamp)}
                    </span>
                    <span className="history-duration">
                      {formatResponseTime(item.responseTime)}
                    </span>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {/* Main Content */}
      <main className="main-content">
        <header>
          <h1>AskNebula</h1>
          <p>A playground to test AskNebula's blockchain query capabilities</p>
        </header>

        <form onSubmit={handleSubmit}>
          <div className="input-container">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter your blockchain query..."
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !query.trim()}>
              {isLoading ? <div className="spinner"></div> : <span>Send</span>}
            </button>
          </div>
        </form>

        {error && (
          <div className="error">
            <p>{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="loading-container">
            <div className="spinner large"></div>
            <p>Processing your query...</p>
          </div>
        )}

        {responses && !isLoading && (
          <div className="responses-container">
            <div className="results-header">
              <h2>Thread Results</h2>
              {responseTime && (
                <div className="response-time">
                  Response time:{" "}
                  <span className="time-value">
                    {formatResponseTime(responseTime)}
                  </span>
                </div>
              )}
            </div>
            {renderContent()}
          </div>
        )}
      </main>

      {/* Overlay that closes the history panel when clicked */}
      {sidebarOpen && (
        <div className="history-overlay" onClick={toggleHistory}></div>
      )}
    </div>
  );
}

export default App;
