import app from './app';
import { config } from './config';
import { connectMongo, disconnectMongo } from './db/mongo';

const PORT = config.port;

async function start(): Promise<void> {
  // Connect to MongoDB before accepting traffic
  if (config.nodeEnv !== 'test') {
    await connectMongo();
  }

  const server = app.listen(PORT, () => {
    console.log(`✓ API Gateway listening on port ${PORT}`);
    console.log(`✓ Environment: ${config.nodeEnv}`);
    console.log(`✓ Health check: http://localhost:${PORT}/api/health`);
  });

  // Graceful shutdown
  async function shutdown(signal: string): Promise<void> {
    console.log(`\n${signal} received — shutting down gracefully`);
    server.close(async () => {
      await disconnectMongo();
      console.log('✓ Server closed');
      process.exit(0);
    });
  }

  process.on('SIGTERM', () => { void shutdown('SIGTERM'); });
  process.on('SIGINT',  () => { void shutdown('SIGINT'); });
}

// Handle uncaught exceptions
process.on('uncaughtException', (error: Error) => {
  console.error('Uncaught Exception:', error);
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', (reason: unknown) => {
  console.error('Unhandled Rejection:', reason);
  process.exit(1);
});

start().catch((err: unknown) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
