# SPDX-FileCopyrightText: 2026 Aevonix Inc and the respective contributors
# SPDX-License-Identifier: AGPL-3.0-only
# Commercial licensing: see COMMERCIAL.md in the FlashAlign repository.
"""A native named provider must resolve to the configured FlashAlign proxy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_setup import wizard
from owner_mode.model_metadata import configure_hermes_provider


class ProviderSetup(unittest.TestCase):
    def config(self):
        return {'model':{'provider':'existing'},'providers':{'existing':{
            'base_url':'http://127.0.0.1:9/v1','default_model':'old',
            'api_key':'synthetic-old-secret','extra_body':{'temperature':0,'max_tokens':512,
                'chat_template_kwargs':{'enable_thinking':False}}}}}

    def test_selects_local_provider_and_keeps_original_definition(self):
        config=self.config();old=copy.deepcopy(config['providers']['existing'])
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1',['text','image'])
        self.assertEqual(config['providers']['existing'],old)
        self.assertEqual(config['model']['provider'],result['name'])
        provider=config['providers'][result['name']]
        self.assertEqual(provider['api_mode'],'chat_completions')
        self.assertEqual(provider['base_url'],'http://127.0.0.1:18034/head/v1')
        self.assertEqual(provider['api_key'],'flashalign-local')
        self.assertEqual(provider['extra_body'],old['extra_body'])
        self.assertTrue(provider['models']['new']['supports_vision'])

    def test_name_collision_does_not_overwrite_an_unmanaged_provider(self):
        config=self.config();config['providers']['flashalign']={'base_url':'preserve'}
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1')
        self.assertEqual(result['name'],'flashalign-2')
        self.assertEqual(config['providers']['flashalign'],{'base_url':'preserve'})

    def test_managed_provider_can_be_reused_after_a_port_change(self):
        config=self.config()
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1')
        config['plugins']={'entries':{'owner-mode':{'settings':{'managed_provider':result['name'],'port':18034}}}}
        again=configure_hermes_provider(config,'newer','http://127.0.0.1:18035/head/v1')
        self.assertEqual(again['name'],result['name'])
        self.assertEqual(config['providers'][again['name']]['default_model'],'newer')
        self.assertEqual(len(config['providers']),2)

    def test_changed_managed_provider_is_preserved_as_user_configuration(self):
        config=self.config()
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1')
        config['plugins']={'entries':{'owner-mode':{'settings':{'managed_provider':result['name'],'port':18034}}}}
        config['providers']['flashalign']['base_url']='http://127.0.0.1:2/v1'
        again=configure_hermes_provider(config,'new','http://127.0.0.1:18035/head/v1')
        self.assertEqual(again['name'],'flashalign-2')
        self.assertEqual(config['providers']['flashalign']['base_url'],'http://127.0.0.1:2/v1')

    def test_provider_body_cannot_pin_the_old_model_or_owner_identity(self):
        config=self.config();extra=config['providers']['existing']['extra_body']
        extra.update(model='old',owner_mode='unrelated',cache_salt='old')
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1')
        copied=config['providers'][result['name']]['extra_body']
        self.assertEqual(copied['max_tokens'],512)
        self.assertNotIn('model',copied);self.assertNotIn('owner_mode',copied);self.assertNotIn('cache_salt',copied)

    def test_nested_provider_credentials_and_routes_are_not_migrated(self):
        config=self.config();extra=config['providers']['existing']['extra_body']
        extra.update(api_key='synthetic',headers={'Authorization':'synthetic'},
                     provider={'route':'another-provider'},messages=[{'content':'synthetic'}],
                     tools=[{'type':'function'}],top_p=.9)
        extra['chat_template_kwargs'].update(headers={'Authorization':'synthetic'},provider={'route':'other'})
        extra['response_format']={'type':'json_object','headers':{'Authorization':'synthetic'}}
        original=copy.deepcopy(config['providers']['existing'])
        result=configure_hermes_provider(config,'new','http://127.0.0.1:18034/head/v1')
        copied=config['providers'][result['name']]['extra_body']
        self.assertEqual(copied,{'max_tokens':512,'temperature':0,'top_p':.9,
                                'chat_template_kwargs':{'enable_thinking':False},
                                'response_format':{'type':'json_object'}})
        self.assertEqual(config['providers']['existing'],original)

    def test_actual_writer_selects_dedicated_provider(self):
        with tempfile.TemporaryDirectory() as temporary:
            home=Path(temporary);path=home/'.hermes/config.yaml';path.parent.mkdir()
            old=self.config();path.write_text(json.dumps(old))
            result=wizard.provision(home,['hermes'],'http://127.0.0.1:12345/v1','new','profile',[],False,
                                    model_info={'input':['text','image']})
            configured=json.loads(path.read_text());name=result['hermes_provider']['name']
            self.assertEqual(configured['providers']['existing'],old['providers']['existing'])
            self.assertEqual(configured['providers'][name]['base_url'],configured['model']['base_url'])
            self.assertEqual(configured['model']['provider'],name)
            self.assertEqual(configured['plugins']['entries']['owner-mode']['settings']['managed_provider'],name)


if __name__=='__main__':unittest.main()
