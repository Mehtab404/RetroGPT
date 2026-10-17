import dotenv from 'dotenv';

dotenv.config();

interface Config {
  port: number;
  nodeEnv: string;
  corsOrigin: string;
  aiServiceUrl: string;
  apiKey: string;
  redisUrl: string;
  mongoUrl: string;
  cacheTtlSeconds: number;
  rateLimitPerMin: number;
}

export const config: Config = {
  port: parseInt(process.env.PORT || '4000', 10),
  nodeEnv: process.env.NODE_ENV || 'development',
  corsOrigin: process.env.CORS_ORIGIN || '*',
  aiServiceUrl: process.env.AI_SERVICE_URL || 'http://ai-service:8000',
  apiKey: process.env.API_KEY || 'change-me',
  redisUrl: process.env.REDIS_URL || 'redis://redis:6379',
  mongoUrl: process.env.MONGO_URL || 'mongodb://mongo:27017/retrogpt',
  cacheTtlSeconds: parseInt(process.env.CACHE_TTL_SECONDS || '86400', 10),
  rateLimitPerMin: parseInt(process.env.RATE_LIMIT_PER_MIN || '30', 10),
};
