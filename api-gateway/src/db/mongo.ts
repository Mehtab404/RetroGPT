/**
 * MongoDB connection management.
 *
 * Uses Mongoose to connect to MongoDB and handles graceful shutdown
 * so in-flight operations can complete before the process exits.
 */

import mongoose from 'mongoose';
import { config } from '../config';
import { logger } from '../utils/logger';

// ---------------------------------------------------------------------------
// Connection state helpers
// ---------------------------------------------------------------------------

/** Returns true when Mongoose is connected or connecting. */
export function isConnected(): boolean {
  return (
    mongoose.connection.readyState === 1 || // connected
    mongoose.connection.readyState === 2 // connecting
  );
}

// ---------------------------------------------------------------------------
// Connect
// ---------------------------------------------------------------------------

/**
 * Connect to MongoDB using the URL from config.
 * Safe to call multiple times — no-ops if already connected.
 */
export async function connectMongo(uri: string = config.mongoUrl): Promise<void> {
  if (isConnected()) return;

  await mongoose.connect(uri, {
    serverSelectionTimeoutMS: 5_000, // fail fast in tests / dev
    socketTimeoutMS: 45_000,
  });

  mongoose.connection.on('error', (err) => {
    logger.error('mongo_connection_error', { error: err });
  });

  mongoose.connection.once('disconnected', () => {
    logger.info('mongo_disconnected');
  });

  logger.info('mongo_connected', { host: mongoose.connection.host });
}

// ---------------------------------------------------------------------------
// Disconnect (graceful shutdown)
// ---------------------------------------------------------------------------

/**
 * Close the Mongoose connection cleanly.
 * Call this on SIGTERM / SIGINT to allow in-flight operations to finish.
 */
export async function disconnectMongo(): Promise<void> {
  if (!isConnected()) return;
  await mongoose.connection.close();
  logger.info('mongo_connection_closed');
}

// ---------------------------------------------------------------------------
// Export the Mongoose instance so callers can access models / transactions
// ---------------------------------------------------------------------------

export { mongoose };
