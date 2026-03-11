#!/bin/bash
echo "🛑 Stopping Factory 5.0 Workcell..."
pkill -f "gz sim"           2>/dev/null || true
pkill -f parameter_bridge   2>/dev/null || true
pkill -f all_transforms     2>/dev/null || true
pkill -f foxglove_bridge    2>/dev/null || true
pkill -f static_transform   2>/dev/null || true
echo "✅ All processes stopped!"
