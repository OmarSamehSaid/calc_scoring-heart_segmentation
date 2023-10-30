 document.addEventListener("DOMContentLoaded", function () {
    // Attach an event listener to the submit button
    document.getElementById("submit-button").addEventListener("click", function () {
        // Create a FormData object to send the file
        var formData = new FormData();
        var fileInput = document.getElementById("dicom_file");
        formData.append("dicom_file", fileInput.files[0]);

        // Send an AJAX POST request to your Flask server
        var xhr = new XMLHttpRequest();
        xhr.open("POST", "/calculate_agatston", true);
        xhr.onreadystatechange = function () {
            if (xhr.readyState === 4) {
                if (xhr.status === 200) {
                    // Parse the JSON response
                    var response = JSON.parse(xhr.responseText);

                    // Update the result container with the Agatston Score
                    var resultContainer = document.getElementById("result-container");
                    resultContainer.innerHTML = "Agatston Score: " + response.agatston_score;

                    // Update the img element's src attribute with the base64 image data
                    var imageElement = document.getElementById("dicom-image");
                    imageElement.src = "data:image/png;base64," + response.image_data;

                    var segElement = document.getElementById("dicom-seg");
                    segElement.src = "data:image/png;base64," + response.seg_data;

                    // Show the image element
                    imageElement.style.display = "block";
                    segElement.style.display = "block";

                } else {
                    // Handle errors here
                    console.error("Error:", xhr.status, xhr.statusText);
                }
            }
        };
        xhr.send(formData);
    });
});