// Get the HTML elements
const promptInput = document.getElementById('chat-input');
const resultOutput = document.getElementById('chat-container');
const submitButton = document.getElementById('send-btn');

// Define the API endpoint
const apiUrl = 'http://localhost:5000/get_response';

// Event listener for the submit button
submitButton.addEventListener('click', () => {
  // Get the input prompt value
  const promptText = promptInput.value;

  // Create a JSON object with the prompt input
  const requestData = {
    prompt_input: promptText,
  };

  // Send a POST request to the Flask API
  fetch(apiUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(requestData),
  })
    .then((response) => {
      if (response.ok) {
        return response.json();
      } else {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
    })
    .then((data) => {
      // Display the result in the output element
      resultOutput.textContent = data.full_response;
    })
    .catch((error) => {
      console.error('Fetch error:', error);
      resultOutput.textContent = 'Error occurred while fetching data.';
    });
});