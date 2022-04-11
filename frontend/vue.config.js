module.exports = {
    devServer: {
      proxy: {
        "/api/": {
          target: "http://150.89.233.81:5500",
        }
      }
    }
  };