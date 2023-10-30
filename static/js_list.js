  document.addEventListener("DOMContentLoaded", function () {
     // Attach an event listener to the submit button
     document.getElementById("submit-button").addEventListener("click", function () {
         // Create a FormData object to send the files
         var formData = new FormData();
         var fileInput = document.getElementById("dicom_files");
         for (var i = 0; i < fileInput.files.length; i++) {
             formData.append("dicom_files", fileInput.files[i]);
         }
 
         // Send an AJAX POST request to your Flask server
         var xhr = new XMLHttpRequest();
         xhr.open("POST", "/calculate_agatstons", true);
         xhr.onreadystatechange = function () {
             if (xhr.readyState === 4) {
                 if (xhr.status === 200) {
                     // Parse the JSON response
                     var response = JSON.parse(xhr.responseText);
 
                     // Update the dicom-seg-container with the received segmented images
                     var segContainer = document.getElementById("dicom-seg-container");
                     segContainer.innerHTML = ""; // Clear previous content
                     for (var j = 0; j < response.seg_images.length; j++) {
                         var img = document.createElement("img");
                         img.src = "data:image/png;base64," + response.seg_images[j];
                         img.className = "uploaded-seg";
                         segContainer.appendChild(img);
                     }
 
                     // Update the result container with the total Agatston score
                     var resultContainer = document.getElementById("result-container");
                     resultContainer.innerHTML = "Total Agatston Score: " + response.total_agatston_score;
 
                 } else {
                     // Handle errors here
                     console.error("Error:", xhr.status, xhr.statusText);
                 }
             }
         };
         xhr.send(formData);
     });
 });
