import React, { useState, useEffect } from "react";
function Cam() {
  const AsyncImage = (props) => {
    const [loadedSrc, setLoadedSrc] = React.useState(null);
    React.useEffect(() => {
      setLoadedSrc(null);
      if (props.src) {
        const handleLoad = () => {
          setLoadedSrc(props.src);
        };
        const image = new Image();
        image.addEventListener("load", handleLoad);
        image.src = props.src;
        return () => {
          image.removeEventListener("load", handleLoad);
        };
      }
    }, [props.src]);
    if (loadedSrc === props.src) {
      return <img {...props} />;
    }
    return null;
  };
  return (
    <div>
      <AsyncImage src="http://localhost:5002/video_feed" alt="Video" />
    </div>
  );
}
export default Cam;
