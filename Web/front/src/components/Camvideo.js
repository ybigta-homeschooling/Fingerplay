import React from "react";
import "./components.css";
function Camvideo({ Id }) {
  const videoData = {
    1: "rudolph2.mov",
    2: "bear2.mov",
    3: "rabbit2.mov",
    4: "butterfly2.mov",
  };
  return (
    <div className="camVideo">
      <video controls="controls">
        <source
          src={require(`../video/${videoData[Id]}#t=5,10`)}
          type="video/mp4"
        />
      </video>
    </div>
  );
}

export default Camvideo;
