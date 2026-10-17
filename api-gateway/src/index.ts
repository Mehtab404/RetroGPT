import app from './app';
import { config } from './config';

const PORT = config.port;

app.listen(PORT, () => {
  console.log(`✓ API Gateway listening on port ${PORT}`);
  console.log(`✓ Environment: ${config.nodeEnv}`);
  console.log(`✓ Health check: http://localhost:${PORT}/api/health`);
});

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
