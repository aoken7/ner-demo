module.exports = {
    devServer: {
      proxy: {
        "/api/": {
          target: "ner-demo_backend://backend:5500",
        }
      }
    }
  };