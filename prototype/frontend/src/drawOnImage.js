export function drawOnImage(image = null) {

    let brushSize
    

    let brushColor
    

    const canvasElement = document.getElementById("canvas");

    const context = canvasElement.getContext("2d");

    const sizeElement = document.getElementById("size-range");
    if (image) {
        // console.log(image)
      const imageWidth = image.width;
      const imageHeight = image.height;
      // rescaling the canvas element
      canvasElement.width = imageWidth;
      canvasElement.height = imageHeight;
      context.drawImage(image, 0, 0, imageWidth, imageHeight);
    }
    
    let isDrawing;
    canvasElement.onmousedown = (e) => {

        // var size = document.getElementById("sizeRange");
        // console.log(size)
      isDrawing = true;
      context.beginPath();
      context.lineWidth = brushSize;
      context.strokeStyle = brushColor;
      context.lineJoin = "round";
      context.lineCap = "round";
      var rect = canvasElement.getBoundingClientRect();
      context.moveTo(e.x - rect.left, e.y - rect.top);
    };
    
    canvasElement.onmousemove = (e) => {
      if (isDrawing) {      
        var rect = canvasElement.getBoundingClientRect();
        context.lineTo(e.x - rect.left, e.y - rect.top);
        context.stroke();      
      }
    };
    
    canvasElement.onmouseup = function () {
      isDrawing = false;
      context.closePath();
    };


  }