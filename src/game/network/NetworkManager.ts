import Peer, { DataConnection } from 'peerjs';
import type { PlayerId } from '@/types/chess';
import type { ChessCommand, NetworkGameSnapshot, NetworkMessage, NetworkRole, LobbyPlayer } from './NetworkTypes';

// Generate a short room code (4 characters)
function generateRoomCode(): string {
  const chars = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'; // Exclude confusing chars
  let code = '';
  for (let i = 0; i < 4; i++) {
    code += chars[Math.floor(Math.random() * chars.length)];
  }
  return code;
}

export class NetworkManager {
  private peer: Peer | null = null;
  private connections: Map<PlayerId, DataConnection> = new Map();
  private role: NetworkRole | null = null;
  private roomCode: string = '';
  private localPlayerId: PlayerId = 1;
  private nextPlayerId: PlayerId = 2;
  private players: Map<PlayerId, LobbyPlayer> = new Map();
  private gameStarted: boolean = false;

  // Callbacks
  public onPlayerJoined?: (player: LobbyPlayer) => void;
  public onPlayerLeft?: (playerId: PlayerId) => void;
  public onStateReceived?: (state: NetworkGameSnapshot) => void;
  public onCommandReceived?: (command: ChessCommand, fromPlayerId: PlayerId) => void;
  public onGameStart?: (playerIds: PlayerId[]) => void;
  public onPlayerAssignment?: (playerId: PlayerId) => void;
  public onError?: (error: string) => void;
  public onConnected?: () => void;

  // Host a new game
  async hostGame(): Promise<string> {
    this.roomCode = generateRoomCode();
    this.role = 'host';
    this.localPlayerId = 1;
    this.nextPlayerId = 2;
    this.players.clear();
    this.connections.clear();

    // Add host as player 1 (white)
    const hostPlayer: LobbyPlayer = {
      playerId: 1,
      name: 'White',
      isHost: true,
    };
    this.players.set(1, hostPlayer);

    return new Promise((resolve, reject) => {
      let resolved = false;

      const timeout = setTimeout(() => {
        if (!resolved) {
          resolved = true;
          this.peer?.destroy();
          reject(new Error('Connection timeout - signaling server may be unavailable'));
        }
      }, 10000);

      this.peer = new Peer(`chess-${this.roomCode}`, {
        debug: 0,
        config: {
          iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
          ],
        },
      });

      this.peer.on('open', () => {
        if (resolved) return;
        resolved = true;
        clearTimeout(timeout);
        console.log('Host peer opened with ID:', this.peer?.id);
        resolve(this.roomCode);
      });

      this.peer.on('connection', (conn) => {
        this.handleIncomingConnection(conn);
      });

      this.peer.on('disconnected', () => {
        console.log('Disconnected from signaling server (normal for P2P)');
      });

      this.peer.on('error', (err) => {
        console.error('Peer error:', err);
        if (err.type === 'unavailable-id') {
          this.peer?.destroy();
          this.roomCode = generateRoomCode();
          this.peer = new Peer(`chess-${this.roomCode}`, {
            debug: 0,
            config: {
              iceServers: [
                { urls: 'stun:stun.l.google.com:19302' },
                { urls: 'stun:stun1.l.google.com:19302' },
              ],
            },
          });
          this.peer.on('open', () => {
            if (resolved) return;
            resolved = true;
            clearTimeout(timeout);
            resolve(this.roomCode);
          });
          this.peer.on('connection', (conn) => this.handleIncomingConnection(conn));
          this.peer.on('disconnected', () => {
            console.log('Disconnected from signaling server');
          });
          this.peer.on('error', (e) => {
            if (resolved) return;
            resolved = true;
            clearTimeout(timeout);
            reject(e);
          });
        } else if (err.type === 'disconnected' || err.type === 'network' || err.type === 'server-error' || err.type === 'socket-error' || err.type === 'socket-closed') {
          if (!resolved) {
            resolved = true;
            clearTimeout(timeout);
            reject(new Error('Could not connect to game server. Please try again.'));
          }
        } else {
          if (resolved) return;
          resolved = true;
          clearTimeout(timeout);
          this.onError?.(err.message);
          reject(err);
        }
      });
    });
  }

  // Join an existing game
  async joinGame(roomCode: string): Promise<void> {
    this.roomCode = roomCode.toUpperCase();
    this.role = 'client';
    this.players.clear();
    this.connections.clear();

    return new Promise((resolve, reject) => {
      const clientId = `chess-client-${Math.random().toString(36).substring(2, 10)}`;
      this.peer = new Peer(clientId, {
        debug: 0,
        config: {
          iceServers: [
            { urls: 'stun:stun.l.google.com:19302' },
            { urls: 'stun:stun1.l.google.com:19302' },
          ],
        },
      });

      this.peer.on('open', () => {
        console.log('Client peer opened, connecting to host...');

        const conn = this.peer!.connect(`chess-${this.roomCode}`, {
          reliable: true,
        });

        conn.on('open', () => {
          console.log('Connected to host');
          this.connections.set(1, conn);
          this.setupConnectionHandlers(conn, 1);
          this.onConnected?.();
          resolve();
        });

        conn.on('error', (err) => {
          console.error('Connection error:', err);
          this.onError?.('Failed to connect to host');
          reject(err);
        });
      });

      this.peer.on('disconnected', () => {
        console.log('Client disconnected from signaling server (P2P still works)');
      });

      this.peer.on('error', (err) => {
        console.error('Peer error:', err);
        if (err.type === 'disconnected' || err.type === 'network') {
          console.log('Signaling server issue (P2P connections still work)');
          return;
        }
        if (err.type === 'peer-unavailable') {
          this.onError?.('Game not found - check the code and try again');
          reject(new Error('Game not found'));
          return;
        }
        this.onError?.(err.message);
        reject(err);
      });

      setTimeout(() => {
        if (this.connections.size === 0) {
          reject(new Error('Connection timeout - room may not exist'));
        }
      }, 10000);
    });
  }

  // Handle incoming connection (host only)
  private handleIncomingConnection(conn: DataConnection): void {
    if (this.gameStarted) {
      conn.close();
      return;
    }

    // Chess is 2 players only
    if (this.nextPlayerId > 2) {
      conn.close();
      return;
    }

    const playerId = this.nextPlayerId as PlayerId;
    this.nextPlayerId = (this.nextPlayerId + 1) as PlayerId;
    this.connections.set(playerId, conn);

    conn.on('open', () => {
      console.log(`Player ${playerId} connected`);

      const playerName = playerId === 1 ? 'White' : 'Black';

      const player: LobbyPlayer = {
        playerId,
        name: playerName,
        isHost: false,
      };
      this.players.set(playerId, player);

      // Send player their assignment
      this.sendTo(playerId, { type: 'playerAssignment', playerId });

      // Send current player list to new player
      for (const p of this.players.values()) {
        this.sendTo(playerId, {
          type: 'playerJoined',
          playerId: p.playerId,
          playerName: p.name,
        });
      }

      // Notify all players about new player
      this.broadcast({ type: 'playerJoined', playerId, playerName });
      this.onPlayerJoined?.(player);

      this.setupConnectionHandlers(conn, playerId);
    });
  }

  // Setup handlers for a connection
  private setupConnectionHandlers(conn: DataConnection, playerId: PlayerId): void {
    conn.on('data', (data) => {
      const message = data as NetworkMessage;
      this.handleMessage(message, playerId);
    });

    conn.on('close', () => {
      console.warn(`[NET] Player ${playerId} connection CLOSED (role=${this.role})`);
      this.connections.delete(playerId);
      this.players.delete(playerId);
      this.onPlayerLeft?.(playerId);

      if (this.role === 'host') {
        this.broadcast({ type: 'playerLeft', playerId });
      }
    });

    conn.on('error', (err) => {
      console.error(`[NET] Connection error with player ${playerId}:`, err);
    });

    const dc = conn.dataChannel;
    if (dc) {
      this.monitorDataChannel(dc, playerId);
    } else {
      let dcAttempts = 0;
      const checkDc = setInterval(() => {
        dcAttempts++;
        if (conn.dataChannel) {
          this.monitorDataChannel(conn.dataChannel, playerId);
          clearInterval(checkDc);
        } else if (dcAttempts > 50) {
          clearInterval(checkDc);
        }
      }, 100);
    }
  }

  private monitorDataChannel(dc: RTCDataChannel, playerId: PlayerId): void {
    dc.addEventListener('close', () => {
      console.warn(`[NET] DataChannel CLOSED for player ${playerId} (state=${dc.readyState})`);
    });
    dc.addEventListener('error', (e) => {
      console.error(`[NET] DataChannel ERROR for player ${playerId}:`, e);
    });
  }

  // Handle incoming message
  private handleMessage(message: NetworkMessage, fromPlayerId: PlayerId): void {
    switch (message.type) {
      case 'state':
        if (this.role === 'client') {
          const state: NetworkGameSnapshot = typeof message.data === 'string'
            ? JSON.parse(message.data)
            : message.data;
          this.onStateReceived?.(state);
        }
        break;

      case 'command':
        if (this.role === 'host') {
          this.onCommandReceived?.(message.data, fromPlayerId);
        }
        break;

      case 'playerAssignment':
        if (this.role === 'client') {
          this.localPlayerId = message.playerId;
          this.onPlayerAssignment?.(message.playerId);
        }
        break;

      case 'gameStart':
        if (this.role === 'client') {
          this.gameStarted = true;
          this.onGameStart?.(message.playerIds);
        }
        break;

      case 'playerJoined':
        this.players.set(message.playerId, {
          playerId: message.playerId,
          name: message.playerName,
          isHost: message.playerId === 1,
        });
        this.onPlayerJoined?.(this.players.get(message.playerId)!);
        break;

      case 'playerLeft':
        this.players.delete(message.playerId);
        this.onPlayerLeft?.(message.playerId);
        break;
    }
  }

  // Send message to specific player (host only)
  private sendTo(playerId: PlayerId, message: NetworkMessage): void {
    const conn = this.connections.get(playerId);
    if (conn && conn.open) {
      conn.send(message);
    }
  }

  // Broadcast message to all connected players (host only)
  private broadcast(message: NetworkMessage): void {
    for (const [, conn] of this.connections) {
      if (conn.open) {
        conn.send(message);
      }
    }
  }

  // Send game state to all clients (host only)
  broadcastState(state: NetworkGameSnapshot): void {
    if (this.role !== 'host') return;
    const jsonString = JSON.stringify(state);
    this.broadcast({ type: 'state', data: jsonString });
  }

  // Send command to host (client only)
  sendCommand(command: ChessCommand): void {
    if (this.role !== 'client') return;
    const hostConn = this.connections.get(1);
    if (hostConn && hostConn.open) {
      hostConn.send({ type: 'command', data: command });
    }
  }

  // Start the game (host only)
  startGame(): void {
    if (this.role !== 'host') return;
    this.gameStarted = true;

    const playerIds = Array.from(this.players.keys()).sort((a, b) => a - b) as PlayerId[];
    this.broadcast({ type: 'gameStart', playerIds });
    this.onGameStart?.(playerIds);
  }

  // Getters
  getRole(): NetworkRole | null {
    return this.role;
  }

  getRoomCode(): string {
    return this.roomCode;
  }

  getLocalPlayerId(): PlayerId {
    return this.localPlayerId;
  }

  getPlayers(): LobbyPlayer[] {
    return Array.from(this.players.values());
  }

  getPlayerCount(): number {
    return this.players.size;
  }

  isHost(): boolean {
    return this.role === 'host';
  }

  isGameStarted(): boolean {
    return this.gameStarted;
  }

  // Disconnect and cleanup
  disconnect(): void {
    for (const conn of this.connections.values()) {
      conn.close();
    }
    this.connections.clear();
    this.peer?.destroy();
    this.peer = null;
    this.role = null;
    this.gameStarted = false;
    this.players.clear();

    this.onPlayerJoined = undefined;
    this.onPlayerLeft = undefined;
    this.onStateReceived = undefined;
    this.onCommandReceived = undefined;
    this.onGameStart = undefined;
    this.onPlayerAssignment = undefined;
    this.onError = undefined;
    this.onConnected = undefined;
  }
}

// Singleton instance
export const networkManager = new NetworkManager();
