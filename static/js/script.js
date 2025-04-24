function adjustImageRatio() {
    const sliderValue = document.getElementById("ratio-slider").value;
    const imgA = document.getElementById("image-a");
    const imgB = document.getElementById("image-b");

    const ratioA = sliderValue;
    const ratioB = 100 - sliderValue;

    imgA.style.flex = ratioA;
    imgB.style.flex = ratioB;
}

function runForwardPass() {
    const sliderValue = parseInt(document.getElementById("ratio-slider").value);
  
    fetch("/run_simulation", {
      method: "POST",
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ slider_value: sliderValue })
    })
    .then(res => res.json())
    .then(() => {
      // Reload column 3 image (simulation)
      document.getElementById("sim-output").src = "/static/img/generated.png?t=" + new Date().getTime();
  
      // ✅ Reload column 4 image (accumulated plot)
      document.getElementById("accumulated-output").src = "/static/img/accumulated.png?t=" + new Date().getTime();
    });
  }
  
  
  function refreshPlot() {
    fetch("/reset_accumulated", { method: "POST" })
      .then(() => {
        document.getElementById("accumulated-output").src = "/static/img/accumulated.png?t=" + new Date().getTime();
      });
  }
  
  