/**
 * Mongoose schema and model for chat sessions.
 *
 * Each session document stores an array of messages in chronological order.
 * Messages can be either from the user or the assistant; assistant messages
 * optionally include source citations returned by the AI service.
 */

import mongoose, { Schema, Document, Model } from 'mongoose';

// ---------------------------------------------------------------------------
// Sub-document types
// ---------------------------------------------------------------------------

export interface SourceDocEmbedded {
  id: string;
  title: string;
  score: number;
  snippet: string;
  page?: number;
}

export interface MessageDoc {
  role: 'user' | 'assistant';
  content: string;
  sources?: SourceDocEmbedded[];
  createdAt: Date;
}

// ---------------------------------------------------------------------------
// Session document interface
// ---------------------------------------------------------------------------

export interface SessionDoc extends Document {
  sessionId: string;
  messages: MessageDoc[];
  createdAt: Date;
  updatedAt: Date;
}

// ---------------------------------------------------------------------------
// Sub-schemas
// ---------------------------------------------------------------------------

const sourceDocSchema = new Schema<SourceDocEmbedded>(
  {
    id: { type: String, required: true },
    title: { type: String, required: true },
    score: { type: Number, required: true },
    snippet: { type: String, required: true },
    page: { type: Number },
  },
  { _id: false }, // no separate _id for embedded source docs
);

const messageSchema = new Schema<MessageDoc>(
  {
    role: {
      type: String,
      enum: ['user', 'assistant'],
      required: true,
    },
    content: { type: String, required: true },
    sources: { type: [sourceDocSchema], default: undefined },
    createdAt: { type: Date, default: Date.now },
  },
  { _id: false }, // keep message array lean — no separate _id per message
);

// ---------------------------------------------------------------------------
// Session schema
// ---------------------------------------------------------------------------

const sessionSchema = new Schema<SessionDoc>(
  {
    sessionId: {
      type: String,
      required: true,
      unique: true,
      index: true,
    },
    messages: { type: [messageSchema], default: [] },
  },
  {
    timestamps: true, // adds createdAt + updatedAt at the document level
    collection: 'sessions',
  },
);

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

/**
 * Retrieve an existing compiled model or compile a new one.
 * This guard prevents "Cannot overwrite model once compiled" errors when the
 * module is re-imported (common in Jest's module isolation).
 */
export const Session: Model<SessionDoc> =
  (mongoose.models['Session'] as Model<SessionDoc> | undefined) ??
  mongoose.model<SessionDoc>('Session', sessionSchema);
