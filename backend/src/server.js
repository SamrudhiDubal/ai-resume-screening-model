'use strict';

const { createApp } = require('./app');

const PORT = process.env.PORT || 5000;
createApp().listen(PORT, () => {
  console.log(`Resume screening API listening on http://localhost:${PORT}`);
});
