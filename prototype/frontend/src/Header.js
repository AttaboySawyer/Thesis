import { AppBar, Toolbar, Typography, makeStyles  } from "@material-ui/core";
import React from "react";

const useStyles = makeStyles(() => ({
    header: {
      backgroundColor: "purple",
      display: 'flex'
    },
    logo: {
      fontFamily: "Work Sans, sans-serif",
      fontWeight: 200,
      color: "#FFFEFE",
      textAlign: "left",
    },
  }));

export default function Header() {
  const { header, logo } = useStyles();

  const displayDesktop = () => {
    return <Toolbar>{snnHelathLogo}</Toolbar>;
  };

  const snnHelathLogo = (
    <Typography variant="h5" component="h1" className={logo}>
      SNNHealth
    </Typography>
  );
  
  return (
    <header>
      <AppBar className={header}>{displayDesktop()}</AppBar>
    </header>
  );
}