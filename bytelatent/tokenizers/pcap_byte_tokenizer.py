# Copyright (c) Meta Platforms, Inc. and affiliates.
"""
A BLT-compatible byte-level PCAP tokenizer with special tokens and packet separation.
"""

from pathlib import Path
from typing import List, Union

import dpkt

from bytelatent.tokenizers.abstract_tokenizer import Tokenizer
from bytelatent.tokenizers.constants import (
    BOE_ID,
    BOS_ID,
    BPE_ID,
    BYTE_UNITS,
    EOS_ID,
    OFFSET,
    PAD_ID,
    PKT_ID,
)


class BltPcapTokenizer(Tokenizer):
    """
    A BLT-compatible byte-level tokenizer for raw network packet data from PCAP files.

    This tokenizer treats each byte (0-255) as a distinct token, offset by OFFSET.
    It includes BLT standard special tokens plus a <pkt> token to separate individual packets.
    
    Token ID mapping:
    - BOE_ID: 0 (Beginning of Example)
    - BOS_ID: 1 (Beginning of Sequence) 
    - EOS_ID: 2 (End of Sequence)
    - BPE_ID: 3 (BPE delimiter)
    - PKT_ID: 4 (Packet separator)
    - Byte values: 5-260 (OFFSET + byte_value)

    Uses dpkt to read the original captured packet bytes and inserts <pkt> tokens
    between individual packets.
    """

    def __init__(
        self,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        add_packet_separators: bool = True,
        vocab_size_unit_1: int = BYTE_UNITS,
    ):
        """
        Initialize the BLT PCAP tokenizer.
        
        Args:
            add_bos: Whether to add beginning-of-sequence token by default
            add_eos: Whether to add end-of-sequence token by default
            add_packet_separators: Whether to add packet separator tokens by default
            vocab_size_unit_1: Number of byte units (should be 256)
        """
        self.add_bos = add_bos
        self.add_eos = add_eos
        self.add_packet_separators = add_packet_separators
        self.vocab_size_unit_1 = vocab_size_unit_1
        
        # Special token IDs from BLT constants
        self.boe_id = BOE_ID
        self.bos_id = BOS_ID
        self.eos_id = EOS_ID
        self.pad_id = PAD_ID
        self.bpe_id = BPE_ID
        self.pkt_id = PKT_ID
        self.offset = OFFSET
        
        # Total vocabulary size
        self.n_words = vocab_size_unit_1 + self.offset

    def get_vocab_size(self) -> int:
        """Returns the size of the vocabulary."""
        return self.n_words

    def encode(
        self, 
        text: str, 
        add_bos: bool | None = None, 
        add_eos: bool | None = None
    ) -> List[int]:
        """
        Encode a PCAP file path to token IDs.
        
        Args:
            text: Path to the PCAP file to tokenize
            add_bos: Whether to add beginning-of-sequence token (None uses default)
            add_eos: Whether to add end-of-sequence token (None uses default)
            
        Returns:
            List of token IDs representing the PCAP file
        """
        if add_bos is None:
            add_bos = self.add_bos
        if add_eos is None:
            add_eos = self.add_eos

        # Read packets from PCAP file
        packets = self._read_pcap_packets(text)
        
        tokens = []
        
        # Add beginning-of-sequence token if requested
        if add_bos:
            tokens.append(self.bos_id)

        # Process each packet
        for i, packet_bytes in enumerate(packets):
            # Add packet separator before each packet (except the first)
            if self.add_packet_separators and i > 0:
                tokens.append(self.pkt_id)

            # Convert each byte to token ID with offset
            packet_tokens = [int(byte) + self.offset for byte in packet_bytes]
            tokens.extend(packet_tokens)

        # Add end-of-sequence token if requested  
        if add_eos:
            tokens.append(self.eos_id)

        return tokens

    def decode(self, tokens: List[int], cut_at_eos: bool = False) -> bytes:
        """
        Decode token IDs back to bytes.
        
        Args:
            tokens: List of token IDs to decode
            cut_at_eos: Whether to stop decoding at EOS token
            
        Returns:
            Decoded bytes object
        """
        if cut_at_eos:
            for k, t in enumerate(tokens):
                if t == self.eos_id:
                    tokens = tokens[: k + 1]
                    break

        # Filter out special tokens and convert back to bytes
        byte_values = []
        for tok in tokens:
            # Only include byte tokens (those with offset subtracted >= 0 and < 256)
            if self.offset <= tok < self.offset + self.vocab_size_unit_1:
                byte_values.append(tok - self.offset)
                
        return bytes(byte_values)

    def get_token_offsets(
        self, text: str, tokens: List[int] | None = None
    ) -> tuple[List[str], List[int]]:
        """
        Return the offsets of the tokens in the original text.
        
        For PCAP files, this returns byte positions and corresponding tokens.
        Note: This is primarily used for evaluation purposes.
        
        Args:
            text: Path to the PCAP file
            tokens: Optional pre-computed tokens
            
        Returns:
            Tuple of (token_strings, byte_offsets)
        """
        if tokens is None:
            tokens = self.encode(text, add_bos=False, add_eos=False)
        
        # Read the raw PCAP data to get actual byte positions
        packets = self._read_pcap_packets(text)
        
        token_strings = []
        byte_offsets = []
        current_offset = 0
        token_idx = 0
        
        for packet_idx, packet_bytes in enumerate(packets):
            # Handle packet separator tokens
            if self.add_packet_separators and packet_idx > 0 and token_idx < len(tokens):
                if tokens[token_idx] == self.pkt_id:
                    token_strings.append("<pkt>")
                    byte_offsets.append(current_offset)
                    token_idx += 1
            
            # Handle byte tokens for this packet
            for byte_pos, byte_val in enumerate(packet_bytes):
                if token_idx < len(tokens):
                    expected_token = byte_val + self.offset
                    if tokens[token_idx] == expected_token:
                        token_strings.append(f"<{byte_val:02x}>")
                        byte_offsets.append(current_offset + byte_pos)
                        token_idx += 1
            
            current_offset += len(packet_bytes)
        
        return token_strings, byte_offsets

    @staticmethod
    def _read_pcap_packets(pcap_path: Union[str, Path]) -> List[bytes]:
        """
        Reads a PCAP file and returns a list of individual packet bytes.

        Args:
            pcap_path: The path to the PCAP file.

        Returns:
            A list of bytes objects, one for each packet.

        Raises:
            FileNotFoundError: If the PCAP file does not exist.
            ValueError: If there is an error reading the PCAP file.
        """
        pcap_path = Path(pcap_path)
        if not pcap_path.is_file():
            raise FileNotFoundError(f"PCAP file not found at: {pcap_path}")

        try:
            packets = []
            with open(pcap_path, 'rb') as f:
                pcap = dpkt.pcap.Reader(f)
                # buf contains the raw packet bytes as captured on the wire
                for ts, buf in pcap:
                    packets.append(buf)
            return packets
        except Exception as e:
            raise ValueError(f"Error reading PCAP file {pcap_path}: {e}")

    # Additional utility methods for PCAP-specific functionality
    
    def tokenize_pcap(
        self, 
        pcap_path: Union[str, Path],
        add_packet_separators: bool | None = None,
        add_bos: bool | None = None,
        add_eos: bool | None = None
    ) -> List[int]:
        """
        Convenience method to tokenize a PCAP file with custom options.
        
        Args:
            pcap_path: Path to the PCAP file to tokenize
            add_packet_separators: Whether to add packet separator tokens
            add_bos: Whether to add beginning-of-sequence token
            add_eos: Whether to add end-of-sequence token
            
        Returns:
            List of token IDs
        """
        # Temporarily override settings if specified
        original_separators = self.add_packet_separators
        if add_packet_separators is not None:
            self.add_packet_separators = add_packet_separators
            
        try:
            return self.encode(str(pcap_path), add_bos=add_bos, add_eos=add_eos)
        finally:
            # Restore original setting
            self.add_packet_separators = original_separators

    def read_pcap_packets(self, pcap_path: Union[str, Path]) -> List[bytes]:
        """
        Public method to read individual packets from a PCAP file.
        
        Args:
            pcap_path: Path to the PCAP file to read
            
        Returns:
            List of raw packet bytes
        """
        return self._read_pcap_packets(pcap_path)

    def decode_to_packets(self, tokens: List[int]) -> List[bytes]:
        """
        Decode token IDs back to individual packet bytes.
        
        Args:
            tokens: List of token IDs to decode
            
        Returns:
            List of individual packet bytes
        """
        packets = []
        current_packet = []
        
        for token in tokens:
            if token == self.pkt_id:
                # Packet separator - save current packet and start new one
                if current_packet:
                    packets.append(bytes(current_packet))
                    current_packet = []
            elif self.offset <= token < self.offset + self.vocab_size_unit_1:
                # Byte token
                current_packet.append(token - self.offset)
            # Skip other special tokens
        
        # Add final packet if any bytes remain
        if current_packet:
            packets.append(bytes(current_packet))
            
        return packets

    def get_special_token_mask(self, tokens: List[int]) -> List[bool]:
        """
        Get a mask indicating which tokens are special tokens.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            List of booleans, True for special tokens
        """
        return [tok < self.offset for tok in tokens]