/**
 * Tests for db/mongo.ts
 *
 * Mongoose is mocked so no real database connection is made.
 */

import { EventEmitter } from 'events';

import { logger } from '../utils/logger';

// ---------------------------------------------------------------------------
// Mock Mongoose before importing the module under test
// ---------------------------------------------------------------------------

const mockConnection = new EventEmitter() as NodeJS.EventEmitter & {
  readyState: number;
  host: string;
  close: jest.Mock;
};
mockConnection.readyState = 0;
mockConnection.host = 'localhost';
mockConnection.close = jest.fn().mockResolvedValue(undefined);

const mockConnect = jest.fn().mockResolvedValue(undefined);

jest.mock('mongoose', () => ({
  connect: mockConnect,
  connection: mockConnection,
  models: {},
  model: jest.fn(),
  Schema: jest.fn(() => ({ index: jest.fn() })),
}));

// ---------------------------------------------------------------------------
// Import after mock is set up
// ---------------------------------------------------------------------------

import { connectMongo, disconnectMongo, isConnected } from '../db/mongo';

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('db/mongo', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockConnection.readyState = 0;
    // Remove any listeners added by previous test runs
    mockConnection.removeAllListeners();
  });

  // -----------------------------------------------------------------------
  // isConnected
  // -----------------------------------------------------------------------

  describe('isConnected()', () => {
    it('returns false when readyState is 0 (disconnected)', () => {
      mockConnection.readyState = 0;
      expect(isConnected()).toBe(false);
    });

    it('returns true when readyState is 1 (connected)', () => {
      mockConnection.readyState = 1;
      expect(isConnected()).toBe(true);
    });

    it('returns true when readyState is 2 (connecting)', () => {
      mockConnection.readyState = 2;
      expect(isConnected()).toBe(true);
    });
  });

  // -----------------------------------------------------------------------
  // connectMongo
  // -----------------------------------------------------------------------

  describe('connectMongo()', () => {
    it('calls mongoose.connect with the provided URI', async () => {
      mockConnection.readyState = 0;
      await connectMongo('mongodb://test:27017/db');
      expect(mockConnect).toHaveBeenCalledTimes(1);
      expect(mockConnect).toHaveBeenCalledWith(
        'mongodb://test:27017/db',
        expect.objectContaining({ serverSelectionTimeoutMS: expect.any(Number) }),
      );
    });

    it('is a no-op when already connected (readyState = 1)', async () => {
      mockConnection.readyState = 1;
      await connectMongo('mongodb://test:27017/db');
      expect(mockConnect).not.toHaveBeenCalled();
    });

    it('is a no-op when already connecting (readyState = 2)', async () => {
      mockConnection.readyState = 2;
      await connectMongo('mongodb://test:27017/db');
      expect(mockConnect).not.toHaveBeenCalled();
    });

    it('registers an error event handler on the connection', async () => {
      mockConnection.readyState = 0;
      await connectMongo('mongodb://test:27017/db');
      // Verify the 'error' listener was registered by firing it
      const errorSpy = jest.spyOn(logger, 'error').mockImplementation(() => undefined);
      mockConnection.emit('error', new Error('test error'));
      expect(errorSpy).toHaveBeenCalledWith('mongo_connection_error', { error: expect.any(Error) });
      errorSpy.mockRestore();
    });

    it('registers a disconnected event handler on the connection', async () => {
      mockConnection.readyState = 0;
      await connectMongo('mongodb://test:27017/db');
      // Verify the 'disconnected' listener was registered by firing it
      const infoSpy = jest.spyOn(logger, 'info').mockImplementation(() => undefined);
      mockConnection.emit('disconnected');
      expect(infoSpy).toHaveBeenCalledWith('mongo_disconnected');
      infoSpy.mockRestore();
    });
  });

  // -----------------------------------------------------------------------
  // disconnectMongo
  // -----------------------------------------------------------------------

  describe('disconnectMongo()', () => {
    it('closes the connection when connected', async () => {
      mockConnection.readyState = 1;
      await disconnectMongo();
      expect(mockConnection.close).toHaveBeenCalledTimes(1);
    });

    it('is a no-op when not connected (readyState = 0)', async () => {
      mockConnection.readyState = 0;
      await disconnectMongo();
      expect(mockConnection.close).not.toHaveBeenCalled();
    });
  });
});
