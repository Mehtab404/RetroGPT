import app from './app';
import { config } from './config';
import { connectMongo, disconnectMongo } from './db/mongo';
import { getRedisClient, disconnectRedis } from './services/cache';
import { logger } from './utils/logger';

const PORT = config.port;

async function start(): Promise<void> {
  // Connect to MongoDB and Redis before accepting traffic
  if (config.nodeEnv !== 'test') {
    await connectMongo();
    await getRedisClient();
  }

  const server = app.listen(PORT, () => {
    logger.info('gateway_started', { port: PORT, env: config.nodeEnv });
  });

  // Graceful shutdown
  async function shutdown(signal: string): Promise<void> {
    logger.info('gateway_stopping', { signal });
    server.close(async () => {
      await disconnectMongo();
      await disconnectRedis();
      logger.info('gateway_stopped');
      process.exit(0);
    });
  }

  process.on('SIGTERM', () => {
    void shutdown('SIGTERM');
  });
  process.on('SIGINT', () => {
    void shutdown('SIGINT');
  });
}

// Handle uncaught exceptions
process.on('uncaughtException', (error: Error) => {
  logger.error('uncaught_exception', { error });
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason: unknown) => {
  logger.error('unhandled_rejection', { error: reason });
  process.exit(1);
});

start().catch((err: unknown) => {
  logger.error('gateway_start_failed', { error: err });
  process.exit(1);
});
