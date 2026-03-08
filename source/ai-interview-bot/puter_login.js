#!/usr/bin/env node
"use strict";

const { getAuthToken } = require("@heyputer/puter.js/src/init.cjs");

async function main() {
  const token = await getAuthToken();
  if (!token) {
    throw new Error("No token returned by Puter.");
  }
  process.stdout.write(`${token}\n`);
}

main().catch((err) => {
  process.stderr.write(`${err.message || String(err)}\n`);
  process.exit(1);
});
