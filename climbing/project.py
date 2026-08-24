"""Project manifest: a route, its clips, their keyframes and cache paths."""

import json
import os
import re
import shutil
from datetime import datetime

from . import video

MANIFEST = 'project.json'
VERSION = 1

LABEL_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_-]*$')


class ProjectError(RuntimeError):
    pass


class Clip:
    def __init__(self, label, data):
        self.label = label
        self._data = data

    @property
    def display_name(self):
        return self._data.get('display_name') or self.label

    @property
    def path(self):
        return self._data['path']

    @property
    def fps(self):
        return float(self._data['fps'])

    @property
    def frame_count(self):
        return int(self._data['frame_count'])

    @property
    def duration(self):
        return float(self._data.get('duration') or self.frame_count / self.fps)

    @property
    def size(self):
        return int(self._data['width']), int(self._data['height'])

    @property
    def flip(self):
        return bool(self._data.get('flip', False))

    @property
    def keyframes(self):
        """Ordered ``[(name, seconds)]``."""
        return [(k['name'], float(k['t'])) for k in self._data.get('keyframes', [])]

    @property
    def keyframe_map(self):
        return {name: t for name, t in self.keyframes}

    def set_keyframe(self, name, seconds):
        marks = [k for k in self._data.get('keyframes', []) if k['name'] != name]
        marks.append({'name': name, 't': round(float(seconds), 3)})
        marks.sort(key=lambda k: k['t'])
        self._data['keyframes'] = marks

    def clear_keyframes(self):
        self._data['keyframes'] = []

    def frame_index(self, seconds):
        return max(0, min(self.frame_count - 1, int(round(seconds * self.fps))))

    def to_dict(self):
        return self._data


class Project:
    def __init__(self, root, data):
        self.root = os.path.abspath(root)
        self._data = data

    # ---- lifecycle -----------------------------------------------------
    @classmethod
    def create(cls, root, name, route=None, grade=None, notes=None):
        root = os.path.abspath(root)
        manifest_path = os.path.join(root, MANIFEST)
        if os.path.exists(manifest_path):
            raise ProjectError(f'该目录已经是一个比对项目: {root}')
        os.makedirs(os.path.join(root, 'videos'), exist_ok=True)
        os.makedirs(os.path.join(root, 'cache'), exist_ok=True)
        os.makedirs(os.path.join(root, 'out'), exist_ok=True)
        data = {
            'version': VERSION,
            'name': name,
            'route': {'name': route or name, 'grade': grade or '', 'notes': notes or ''},
            'created_at': datetime.now().isoformat(timespec='seconds'),
            'reference': None,
            'clips': {},
        }
        project = cls(root, data)
        project.save()
        return project

    @classmethod
    def load(cls, root):
        root = os.path.abspath(root)
        manifest_path = os.path.join(root, MANIFEST)
        if not os.path.exists(manifest_path):
            raise ProjectError(f'{root} 下没有 {MANIFEST}，请先运行 `init`')
        with open(manifest_path, encoding='utf-8') as fh:
            data = json.load(fh)
        if data.get('version') != VERSION:
            raise ProjectError(f'项目文件版本不兼容: {data.get("version")}')
        return cls(root, data)

    @classmethod
    def discover(cls, start=None):
        """Walk up from ``start`` looking for a manifest."""
        current = os.path.abspath(start or os.getcwd())
        while True:
            if os.path.exists(os.path.join(current, MANIFEST)):
                return cls.load(current)
            parent = os.path.dirname(current)
            if parent == current:
                raise ProjectError(
                    '当前目录及其上级都没有找到比对项目，'
                    '请用 -p/--project 指定项目目录，或先运行 `init`'
                )
            current = parent

    def save(self):
        path = os.path.join(self.root, MANIFEST)
        tmp = path + '.tmp'
        with open(tmp, 'w', encoding='utf-8') as fh:
            json.dump(self._data, fh, ensure_ascii=False, indent=2)
        os.replace(tmp, path)

    # ---- accessors -----------------------------------------------------
    @property
    def name(self):
        return self._data['name']

    @property
    def route(self):
        return self._data.get('route', {})

    @property
    def reference(self):
        return self._data.get('reference')

    @reference.setter
    def reference(self, label):
        if label is not None and label not in self._data['clips']:
            raise ProjectError(f'没有这条片段: {label}')
        self._data['reference'] = label

    @property
    def clips(self):
        return {label: Clip(label, data) for label, data in self._data['clips'].items()}

    def clip(self, label):
        if label not in self._data['clips']:
            known = ', '.join(self._data['clips']) or '(空)'
            raise ProjectError(f'没有这条片段: {label}。已有片段: {known}')
        return Clip(label, self._data['clips'][label])

    def path(self, *parts):
        return os.path.join(self.root, *parts)

    def abspath(self, relative):
        return relative if os.path.isabs(relative) else os.path.join(self.root, relative)

    # ---- mutation ------------------------------------------------------
    def add_clip(self, label, source, display_name=None, flip=False, copy=True):
        if not LABEL_RE.match(label):
            raise ProjectError('片段标识只能用字母、数字、下划线和短横线，且以字母或数字开头')
        if label in self._data['clips']:
            raise ProjectError(f'片段标识已存在: {label}（用 remove 先删除）')

        source = os.path.abspath(source)
        meta = video.probe(source)

        if copy:
            ext = os.path.splitext(source)[1] or '.mp4'
            target = self.path('videos', f'{label}{ext}')
            if os.path.abspath(target) != source:
                shutil.copy2(source, target)
            rel = os.path.relpath(target, self.root)
        else:
            rel = source

        self._data['clips'][label] = {
            'label': label,
            'display_name': display_name or label,
            'path': rel,
            'flip': bool(flip),
            'keyframes': [],
            'pose': None,
            **meta,
        }
        if self._data.get('reference') is None:
            self._data['reference'] = label
        return self.clip(label)

    def remove_clip(self, label):
        clip = self.clip(label)
        stored = self.abspath(clip.path)
        if os.path.commonpath([stored, self.path('videos')]) == self.path('videos'):
            if os.path.exists(stored):
                os.remove(stored)
        cache = self._data['clips'][label].get('pose')
        if cache and os.path.exists(self.abspath(cache)):
            os.remove(self.abspath(cache))
        del self._data['clips'][label]
        if self._data.get('reference') == label:
            self._data['reference'] = next(iter(self._data['clips']), None)

    def set_pose_cache(self, label, relative_path):
        self._data['clips'][label]['pose'] = relative_path

    def pose_cache_path(self, label):
        cache = self._data['clips'][label].get('pose')
        return self.abspath(cache) if cache else None

    # ---- alignment readiness -------------------------------------------
    def shared_keyframes(self, labels=None):
        """Keyframe names present in every requested clip, in reference order."""
        clips = self.clips
        labels = labels or list(clips)
        if not labels:
            return []
        maps = [clips[label].keyframe_map for label in labels]
        common = set(maps[0])
        for m in maps[1:]:
            common &= set(m)
        ref_label = self.reference if self.reference in labels else labels[0]
        order = [name for name, _ in clips[ref_label].keyframes if name in common]
        return order
