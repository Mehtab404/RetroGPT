import express, { Application, Request, Response } from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import { errorHandler } from './middleware/errorHandler';
import { requireApiKey } from './middleware/auth';
import { rateLimit } from './middleware/rateLimit';
import chatRouter from './routes/chat';
import sessionsRouter from './routes/sessions';

const app: Application = express();

// Behind Nginx in production — trust the first proxy so req.ip reflects the
// real client address (used for per-IP rate limiting).
app.set('trust proxy', 1);

// Security middleware
app.use(helmet());
app.use(
  cors({
    origin: config.corsOrigin,
    credentials: true,
  }),
);

// Body parsing
app.use(express.json({ limit: '1mb' }));
app.use(express.urlencoded({ extended: true, limit: '1mb' }));

// Health check endpoint
app.get('/api/health', async (_req: Request, res: Response) => {
  res.json({
    status: 'ok',
    service: 'api-gateway',
    timestamp: new Date().toISOString(),
  });
});

// Root endpoint
app.get('/', (_req: Request, res: Response) => {
  res.json({
    service: 'RetroGPT API Gateway',
    version: '0.1.0',
    health: '/api/health',
  });
});

// Authenticated, rate-limited API surface.
// `/api/health` is registered above and intentionally stays public so
// container healthchecks and load balancers can probe it without a key.
app.use('/api', rateLimit());
app.use('/api', requireApiKey);
app.use('/api', chatRouter);
app.use('/api', sessionsRouter);

// Error handling middleware (must be last)
app.use(errorHandler);

export default app;
