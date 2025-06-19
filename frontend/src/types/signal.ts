export interface Signal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL';
  entry_price: number;
  target_price: number;
  stop_loss: number;
  confidence: number;
  sentiment_score: number;
  created_at: string;
  status: 'active' | 'completed' | 'stopped';
  risk_reward_ratio?: number;
}

export interface SignalResponse {
  signals: Signal[];
  count: number;
  generated_at: string;
}