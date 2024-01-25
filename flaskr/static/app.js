function handleKeyPress(event) {
    // Check if the pressed key is 'Enter'
    if (event.key === 'Enter') {
      sendMessage()
    }
}
function sendMessage() {
    const userInput = document.getElementById("userInput").value;
    if (userInput.trim() === "") return;

    const chatBody = document.getElementById("chatBody");
    const userMessage = document.createElement("div");
    userMessage.className = "message user-message";
    userMessage.innerHTML = '<span class="message-text">' + userInput + '</span>';
    chatBody.appendChild(userMessage);

    // Show loading indicator while waiting for the response
    showLoading();

    // Send the user message to the server using Ajax
    fetch("/chat-gpt", {
        method: "POST",
        headers: {
            "Content-Type": "application/json",
        },
        body: JSON.stringify({ "user_input": userInput }),
    })
    .then(response => response.json())
    .then(data => {
        var gptMessage = document.createElement("div");
        var geminiMessage = document.createElement("div");
        gptMessage.className = "message bot-message";
        gptMessage.innerHTML = '<span class="message-text">' + 'GPT: ' + data.gpt_message + '</span>';
        chatBody.appendChild(gptMessage);
        
        geminiMessage.className = "message bot-message";
        geminiMessage.innerHTML = '<span class="message-text">' + 'Gemini: ' + data.gemini_message + '</span>';
        chatBody.appendChild(geminiMessage);

        // Scroll to the bottom of the chat body
        chatBody.scrollTop = chatBody.scrollHeight;

        // Hide loading indicator after receiving the response
        hideLoading();
    });

    // Clear the user input field
    document.getElementById("userInput").value = "";
}

function showLoading() {
    document.getElementById("loadingBubble").style.display = "block";
}

function hideLoading() {
    document.getElementById("loadingBubble").style.display = "none";
}

function uploadFile() {
    var fileInput = document.getElementById("fileInput");

    if (fileInput.files.length) {
        // Show loading indicator while waiting for the response
        showLoading();

        // Upload the selected file to the server using Ajax
        var formData = new FormData();
        formData.append("file", fileInput.files[0]);

        fetch("/files", {
            method: "POST",
            body: formData,
        })

        .then(response => response.json())
        .then(data => {
            // Handle the server response 
            console.log("File uploaded:", data);
        })
        .finally(() => {
            // Hide loading indicator after receiving the response
            hideLoading();
        });
    }
}