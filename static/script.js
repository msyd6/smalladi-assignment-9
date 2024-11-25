document.getElementById("experiment-form").addEventListener("submit", async function(event) {
    event.preventDefault(); 

    const activation = document.getElementById("activation").value.toLowerCase();
    const lr = parseFloat(document.getElementById("lr").value);
    const hiddenDim = parseInt(document.getElementById("hidden_dim").value);
    const stepNum = parseInt(document.getElementById("step_num").value);

    // Validation checks
    if (isNaN(lr)) {
        alert("Please enter a valid number for learning rate.");
        return;
    }

    if (isNaN(hiddenDim) || hiddenDim <= 0) {
        alert("Please enter a positive integer for Hidden Layer Size.");
        return;
    }

    if (isNaN(stepNum) || stepNum <= 0) {
        alert("Please enter a positive integer for Number of Training Steps.");
        return;
    }

    fetch("/run_experiment", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ activation: activation, lr: lr, hidden_dim: hiddenDim, step_num: stepNum })
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert("Error: " + data.error);
        } else {
            const resultsDiv = document.getElementById("results");
            resultsDiv.style.display = "block";

            const resultImg = document.getElementById("result_gif");
            if (data.result_gif) {
                const timestamp = new Date().getTime();
                resultImg.src = `/${data.result_gif}?t=${timestamp}`;
                resultImg.style.display = "block";
            } else {
                alert("No result GIF found.");
            }
        }
    })
    .catch(error => {
        console.error("Error running experiment:", error);
        alert("An error occurred while running the experiment.");
    });
});


// Randomize button functionality
document.getElementById("randomize-btn").addEventListener("click", function() {
    document.getElementById("activation").value = ["tanh", "relu", "sigmoid"][Math.floor(Math.random() * 3)];
    document.getElementById("lr").value = (Math.random() * 0.1).toFixed(2);
    document.getElementById("hidden_dim").value = Math.floor(Math.random() * 10) + 1;
    document.getElementById("step_num").value = Math.floor(Math.random() * 1000) + 100;
});

// Clear button functionality
document.getElementById("clear-btn").addEventListener("click", function() {
    document.getElementById("activation").value = "tanh";
    document.getElementById("lr").value = "";
    document.getElementById("hidden_dim").value = "";
    document.getElementById("step_num").value = "";
    document.getElementById("results").style.display = "none";
    document.getElementById("result_gif").style.display = "none";
});
