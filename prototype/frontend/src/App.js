import { Paper, RootRef, Typography, Button , Slider, Divider, FormControl, FormLabel, RadioGroup, FormControlLabel, Radio} from "@material-ui/core";
import './App.css';
import Header from './Header';
import React, { Component } from "react";
import FileControl from "./FileControl";
import { saveAs } from 'file-saver'
import { drawOnImage } from "./drawOnImage";
const axios = require('axios').default;

export class ImageDisplay extends Component {
  constructor(props) {
    super(props);
    this.updateImages = this.updateImages.bind(this)
    this.state = {
      imageArray: <h2>Processed Images will appear here</h2>
    };
  }

  updateImages(images){
    const imageDiv = images.map(img => {
      return <div><img src={`data:image/png;base64,${img}`} alt="" /></div>
    })
    this.setState({ resp: imageDiv });
  }

  render(){
    return(
      <div>
        {this.state.imageArray}
      </div>
    )
  }
}

const download = () => {
  var canvasElement = document.getElementById("canvas");
  
  // let src = document.getElementById('bigImage').src
  console.log(canvasElement.toDataURL())
  if (canvasElement.toDataURL() == 'http://localhost:3000/Placeholder.png') {
    alert("No image to save! Please process and select an image first")
  } else {
    saveAs(canvasElement.toDataURL(), 'test.png')
  }
}

function App() {

  return (
    <div style={{display: 'flex', flexDirection: 'column', height: '100%'}}>
      <div>
      <Typography variant="h4" style={{      
        fontFamily: "Work Sans, sans-serif",
        fontWeight: 300,
        color: "#FFFEFE",
        backgroundColor: 'purple',
        padding: 10}}>SNNHealth</Typography>
      </div>
      <div style={{backgroundColor: '#ECECEC', display: 'flex', flexDirection: 'row', height: '100%'}}>
        
        <div id="file-upload-containter" style={{display: 'flex', flexDirection: 'column', width: '35%'}}>
            <FileControl />
        </div >

        <div id="image-diplay-container" style={{display: 'flex', flexDirection: 'column', width: '40%'}}>
          <Paper elevation={5} style={{marginTop: 85, margin: 20, height: 800, padding: 15, display: 'grid', alignItems:'center', justifyContent:'center'}}>
            <div style={{display:"flex"}}>
              <div style={{width: '100%', height:700, justifyContent: 'center'}}>
                <canvas id="canvas" width="700" height="700"></canvas>
                {/* <img id="bigImage" style={{display: 'flex', margin:5, width:700, height:700, justifySelf:'center'}} src='./Placeholder.png' alt="" /> */}
                <Button id="downloadButton" onClick={download} style={{marginTop:10, backgroundColor:'purple', color:'white'}}>Save Image</Button>
              </div>
            </div>   
          </Paper>
        </div>

        <div id="annotations-container" style={{display: 'flex', flexDirection: 'column', width: '25%'}}>
            <Paper elevation={5} style={{marginTop: 85, margin: 20, height: '60%', padding: 15, display:'grid'}}>
              {/* <p>Annotations capabilities go here?</p> */}
              <div>
                <Typography variant="h5">Annotations</Typography>
                <Divider style={{margin:5}}/>
                <Typography variant="h6">Size</Typography>
                <Slider id="size-range" defaultValue={5} min={1} max={20} valueLabelDisplay="auto" style={{color:"purple"}} 
                onChange={(e,v) => {
                      const canvasElement = document.getElementById("canvas");
                      const context = canvasElement.getContext("2d");
                      context.lineWidth = v
                }}
                ></Slider>
                <Divider style={{margin:5}}/>
                <Typography variant="h6">Color</Typography>
                <FormControl>
                  <RadioGroup
                    defaultValue="black"
                    name="radio-buttons-group"
                    onChange={(e,v) => {
                      const canvasElement = document.getElementById("canvas");
                      const context = canvasElement.getContext("2d");
                      context.strokeStyle = v
                }}
                  >
                    <FormControlLabel value="black" control={<Radio style={{color:'black'}}/>} label="Black" />
                    <FormControlLabel value="red" control={<Radio style={{color:'red'}}/>} label="Red" />
                    <FormControlLabel value="green" control={<Radio style={{color:'green'}}/>} label="Green" />
                    <FormControlLabel value="blue" control={<Radio style={{color:'blue'}}/>} label="Blue" />
                  </RadioGroup>
                </FormControl>
              </div>
              <div style={{alignSelf:"flex-end"}}>
              <Divider style={{margin:5}}/>
              <Button id="clear-button" onClick={() => {
                      const canvasElement = document.getElementById("canvas");
                      const context = canvasElement.getContext("2d");
                      context.clearRect(0, 0, canvasElement.width, canvasElement.height);
              }} style={{marginTop:10, backgroundColor:'purple', color:'white'}}>Clear</Button>
              </div>
            </Paper>
            
            <Paper elevation={5} style={{margin: 20, height: '45%', display: 'grid'}}>
              <textarea style={{borderColor: 'white',width:'95%', height:'90%', alignSelf:'center', justifySelf:'center'}} placeholder="Enter Notes here"/>
            </Paper>
        </div>

      </div>
    </div>
  );
}

export default App;
